import os
import json
import math
import time
import zipfile
from pathlib import Path
import numpy as np
import io
import torch
import torch.nn as nn
import torch.optim as optim
from env.vector_env import SubprocVecEnv

class ActorCritic(nn.Module):
    def __init__(self,obs_dim,act_dim):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(obs_dim,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,act_dim))
        self.log_std=nn.Parameter(torch.zeros(act_dim))
        self.critic=nn.Sequential(nn.Linear(obs_dim,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,1))
    def value(self,x):
        return self.critic(x)
    def dist(self,x):
        mean=self.actor(x)
        std=torch.exp(self.log_std)
        return mean,std
    def act(self,obs):
        mean,std=self.dist(obs)
        z=mean+std*torch.randn_like(mean)
        a=torch.tanh(z)
        return a,z,mean,std
    def log_prob(self,z,mean,std):
        var=std*std
        logp=-0.5*(((z-mean)*(z-mean))/var+2*torch.log(std)+math.log(2*math.pi))
        logp=logp.sum(dim=-1)
        return logp

class RolloutBuffer:
    def __init__(self,capacity,obs_dim,act_dim):
        self.capacity=capacity
        self.ptr=0
        self.obs=np.zeros((capacity,obs_dim),dtype=np.float32)
        self.act=np.zeros((capacity,act_dim),dtype=np.float32)
        self.rew=np.zeros((capacity,),dtype=np.float32)
        self.val=np.zeros((capacity,),dtype=np.float32)
        self.next_val=np.zeros((capacity,),dtype=np.float32)
        self.logp=np.zeros((capacity,),dtype=np.float32)
        self.done=np.zeros((capacity,),dtype=np.float32)
    def store(self,o,a,r,v,nv,lp,d):
        i=self.ptr
        self.obs[i]=o
        self.act[i]=a
        self.rew[i]=r
        self.val[i]=v
        self.next_val[i]=nv
        self.logp[i]=lp
        self.done[i]=d
        self.ptr+=1
    def ready(self):
        return self.ptr>=self.capacity
    def reset(self):
        self.ptr=0

class PPOTrainer:
    def __init__(self,env,cfg):
        self.env=env
        self.cfg=cfg
        self.paths=cfg["paths"]
        self.total_steps=cfg["training"]["total_steps"]
        self.update_interval=cfg["training"]["update_interval"]
        self.batch_size=cfg["training"]["batch_size"]
        self.gamma=cfg["training"]["gamma"]
        self.lam=cfg["training"]["gae_lambda"]
        self.clip=cfg["training"]["clip_range"]
        self.vf_coef=cfg["training"]["vf_coef"]
        self.ent_coef=cfg["training"]["ent_coef"]
        self.lr=cfg["training"]["learning_rate"]
        self.max_grad_norm=cfg["training"]["max_grad_norm"]
        self.ck_interval=cfg["training"]["checkpoint_interval"]
        self.viewer_interval=cfg["training"]["viewer_update_interval"]
        self.opp_refresh=cfg["training"]["opponent_refresh_interval"]
        self.console_interval=float(self.cfg.get("training",{}).get("console_update_interval",5))
        self.pacing=self.cfg.get("training",{}).get("simulation_pace","max")
        if isinstance(self.env, SubprocVecEnv):
            o_blue,_=self.env.reset()
            obs_dim=len(o_blue[0])
            self.num_envs=self.env.n
        else:
            o_blue,_=self.env.reset()
            obs_dim=len(o_blue)
            self.num_envs=1
        act_dim=7
        self.obs_dim=obs_dim
        self.act_dim=act_dim
        self.model=ActorCritic(obs_dim,act_dim)
        self.opt=optim.Adam(self.model.parameters(),lr=self.lr)
        self.buffer=RolloutBuffer(self.update_interval*self.num_envs,obs_dim,act_dim)
        self.iterations=0
        self.n_updates=0
        self.start_time=time.time()
        self.step_count=0
        self.last_viewer_time=0.0
        self.last_print_time=0.0
        self.profile={"env_step":0.0,"model_forward":0.0,"update":0.0,"io":0.0,"loops":0}
        self.reward_history=[]
        self.ep_length_history=[]
        self.cur_ep_steps=0
        Path(self.paths["outputs"]).mkdir(parents=True,exist_ok=True)
        Path(self.paths["checkpoints"]).mkdir(parents=True,exist_ok=True)
        Path(self.paths["final"]).mkdir(parents=True,exist_ok=True)
        Path(self.paths["backups"]).mkdir(parents=True,exist_ok=True)
        self._attempt_resume()
    def train(self):
        if isinstance(self.env, SubprocVecEnv):
            o_blue,_=self.env.reset()
        else:
            o_blue,_=self.env.reset()
        done=False
        while self.step_count<self.total_steps:
            loop_start=time.time()
            if self.num_envs>1:
                obs=torch.tensor(o_blue,dtype=torch.float32)
                with torch.no_grad():
                    mf_start=time.time()
                    a,z,mean,std=self.model.act(obs)
                    v=self.model.value(obs).squeeze(-1)
                    logp=self.model.log_prob(z,mean,std)
                    self.profile["model_forward"]+=time.time()-mf_start
                pa_np=a.numpy()
                es=time.time()
                ob_blue_next, _, r_blue, _, done_arr=self.env.step(pa_np)
                self.profile["env_step"]+=time.time()-es
                next_v=torch.tensor(ob_blue_next,dtype=torch.float32)
                with torch.no_grad():
                    nv=self.model.value(next_v).squeeze(-1)
                for i in range(self.num_envs):
                    self.buffer.store(o_blue[i],pa_np[i],float(r_blue[i]),float(v[i].item()),float(nv[i].item()),float(logp[i].item()),1.0 if done_arr[i] else 0.0)
                    self.reward_history.append(float(r_blue[i]))
                o_blue=ob_blue_next
                self.step_count+=self.num_envs
                self.cur_ep_steps+=1
            else:
                obs=torch.tensor(o_blue,dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    mf_start=time.time()
                    a,z,mean,std=self.model.act(obs)
                    v=self.model.value(obs).item()
                    logp=self.model.log_prob(z,mean,std).item()
                    pa=a.squeeze(0).numpy()
                es=time.time()
                ob_blue_next, _, r_blue, _, done=self.env.self_play_step(pa)
                self.profile["env_step"]+=time.time()-es
                self.profile["model_forward"]+=time.time()-mf_start
                nv=float(self.model.value(torch.tensor(ob_blue_next,dtype=torch.float32).unsqueeze(0)).item())
                self.buffer.store(o_blue,pa,r_blue,v,nv,logp,1.0 if done else 0.0)
                self.reward_history.append(float(r_blue))
                o_blue=ob_blue_next
                self.step_count+=1
                self.cur_ep_steps+=1
            if self.buffer.ready():
                us=time.time()
                self._update()
                self.profile["update"]+=time.time()-us
                self.buffer.reset()
                self.iterations+=1
            if self.step_count%self.ck_interval==0:
                self._save_checkpoint(self.step_count)
                self._backup_latest()
            if self.step_count%self.opp_refresh==0:
                try:
                    self.env._refresh_opponent_policy()
                except Exception:
                    pass
            t=time.time()
            if t-self.last_viewer_time>self.viewer_interval:
                ios=time.time()
                self._write_live_metrics()
                self.profile["io"]+=time.time()-ios
                self.last_viewer_time=t
            if self.pacing=="realtime":
                time.sleep(max(0.0, (1.0/120.0)))
            self.profile["loops"]+=1
            if done:
                o_blue,_=self.env.reset()
                self.ep_length_history.append(self.cur_ep_steps)
                self.cur_ep_steps=0
                done=False
            if t-self.last_print_time>self.console_interval:
                try:
                    self._print_box(self._collect_training_metrics())
                except Exception:
                    pass
                self.last_print_time=t
        self._save_final()
    def _update(self):
        obs=torch.tensor(self.buffer.obs,dtype=torch.float32)
        act=torch.tensor(self.buffer.act,dtype=torch.float32)
        rew=torch.tensor(self.buffer.rew,dtype=torch.float32)
        val=torch.tensor(self.buffer.val,dtype=torch.float32)
        nval=torch.tensor(self.buffer.next_val,dtype=torch.float32)
        done=torch.tensor(self.buffer.done,dtype=torch.float32)
        adv=self._gae(rew,val,nval,done,self.gamma,self.lam)
        # normalize advantages for stability
        adv=(adv-adv.mean())/(adv.std()+1e-8)
        ret=adv+val
        old_logp=torch.tensor(self.buffer.logp,dtype=torch.float32)
        for _ in range(int(max(1,(self.update_interval*self.num_envs)//self.batch_size))):
            idx=np.random.randint(0,self.buffer.ptr,self.batch_size)
            bobs=obs[idx]
            bact=act[idx]
            badv=adv[idx]
            bret=ret[idx]
            blogp_old=old_logp[idx]
            mean,std=self.model.dist(bobs)
            z=torch.atanh(torch.clamp(bact,-0.999,0.999))
            blogp=self.model.log_prob(z,mean,std)
            ratio=torch.exp(blogp-blogp_old)
            pg1=badv*ratio
            pg2=badv*torch.clamp(ratio,1.0-self.clip,1.0+self.clip)
            pg_loss=-torch.mean(torch.min(pg1,pg2))
            clip_mask=((ratio>1.0+self.clip) | (ratio<1.0-self.clip)).float()
            clip_frac=clip_mask.mean().item()
            vpred=self.model.value(bobs).squeeze(-1)
            vf_loss=torch.mean((vpred-bret)**2)
            ent=torch.mean(torch.log(std))
            loss=pg_loss+self.vf_coef*vf_loss-self.ent_coef*ent
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),self.max_grad_norm)
            self.opt.step()
            self.n_updates+=1
        self._last_clip_fraction=clip_frac
    def _gae(self,rew,val,nval,done,gamma,lam):
        T=len(rew)
        adv=torch.zeros(T)
        last=0.0
        for t in reversed(range(T)):
            mask=1.0-done[t]
            delta=rew[t]+gamma*mask*nval[t]-val[t]
            last=delta+gamma*lam*mask*last
            adv[t]=last
        return adv
    def _checkpoint_name(self,step):
        return f"step_{step}_ppo.zip"
    def _save_checkpoint(self,step):
        name=self._checkpoint_name(step)
        path=os.path.join(self.paths["checkpoints"],name)
        tmp=os.path.join(self.paths["checkpoints"],f"tmp_{name}")
        try:
            with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
                buf=io.BytesIO()
                torch.save(self.model.state_dict(),buf)
                z.writestr("model.pt",buf.getvalue())
                obuf=io.BytesIO()
                torch.save(self.opt.state_dict(),obuf)
                z.writestr("optimizer.pt",obuf.getvalue())
                z.writestr("meta.json",json.dumps({"step":step,"obs_dim":self.obs_dim,"act_dim":self.act_dim}).encode("utf-8"))
            os.replace(tmp,path)
        except Exception:
            pass
    def _save_final(self):
        name=f"final_ppo.zip"
        path=os.path.join(self.paths["final"],name)
        tmp=os.path.join(self.paths["final"],f"tmp_{name}")
        try:
            with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
                buf=io.BytesIO()
                torch.save(self.model.state_dict(),buf)
                z.writestr("model.pt",buf.getvalue())
                obuf=io.BytesIO()
                torch.save(self.opt.state_dict(),obuf)
                z.writestr("optimizer.pt",obuf.getvalue())
                z.writestr("meta.json",json.dumps({"step":self.step_count,"obs_dim":self.obs_dim,"act_dim":self.act_dim}).encode("utf-8"))
            os.replace(tmp,path)
            try:
                self._backup_latest()
            except Exception:
                pass
        except Exception:
            pass
    def _write_live_metrics(self):
        try:
            touches=sum(1 for r in self.reward_history[-int(self.viewer_interval*self.update_interval*self.num_envs):] if r>0.0)
            elapsed=time.time()-self.start_time
            fps=float((self.step_count)/elapsed) if elapsed>0 else 0.0
            data={"step":self.step_count,"recent_reward":float(np.mean(self.buffer.rew[:self.buffer.ptr])) if self.buffer.ptr>0 else 0.0,"episode_time":float(getattr(self.env,'episode_time',0.0)),"reward_history":self.reward_history[-200:],"episode_length_history":self.ep_length_history[-50:],"touches_recent":touches,"iterations":self.iterations,"time_elapsed":int(elapsed),"total_timesteps":self.step_count,"fps":int(fps),"n_updates":self.n_updates,"learning_rate":self.lr,"clip_range":self.clip,"clip_fraction":float(getattr(self,'_last_clip_fraction',0.0))}
            with open(self.paths["live_metrics"],"w") as f:
                json.dump(data,f)
            with open(os.path.join(self.paths["outputs"],"profile.json"),"w") as pf:
                json.dump(self.profile,pf)
        except Exception:
            pass
    def _attempt_resume(self):
        try:
            mode=self.cfg.get("training",{}).get("resume_mode","auto")
            final_dir=self.paths["final"]
            ck_dir=self.paths["checkpoints"]
            chosen=None
            if mode in ("final","auto"):
                fp=os.path.join(final_dir,"final_ppo.zip")
                if os.path.exists(fp):
                    chosen=fp
            if chosen is None and mode in ("checkpoint","auto"):
                files=[f for f in os.listdir(ck_dir) if f.endswith("_ppo.zip")]
                if files:
                    files_sorted=sorted(files)
                    chosen=os.path.join(ck_dir,files_sorted[-1])
            if chosen is None:
                return
            import zipfile, io
            with zipfile.ZipFile(chosen,"r") as z:
                if "model.pt" in z.namelist():
                    m=z.read("model.pt")
                    self.model.load_state_dict(torch.load(io.BytesIO(m),map_location="cpu"))
                if "optimizer.pt" in z.namelist():
                    o=z.read("optimizer.pt")
                    self.opt.load_state_dict(torch.load(io.BytesIO(o),map_location="cpu"))
            meta_step=0
            try:
                with zipfile.ZipFile(chosen,"r") as z:
                    meta=json.loads(z.read("meta.json"))
                    meta_step=int(meta.get("step",0))
            except Exception:
                pass
            self.step_count=max(self.step_count,meta_step)
        except Exception:
            pass
    def _backup_latest(self):
        try:
            src=os.path.join(self.paths["final"],"final_ppo.zip")
            if not os.path.exists(src):
                return
            tgt=os.path.join(self.paths["backups"],f"backup_{int(time.time())}_final_ppo.zip")
            with open(src,"rb") as s:
                with open(tgt,"wb") as t:
                    t.write(s.read())
        except Exception:
            pass
    def _collect_training_metrics(self):
        if self.buffer.ptr>0:
            try:
                vals=torch.tensor(self.buffer.val[:self.buffer.ptr])
                rew=torch.tensor(self.buffer.rew[:self.buffer.ptr])
                done=torch.tensor(self.buffer.done[:self.buffer.ptr])
                adv=self._gae(rew,vals,done,self.gamma,self.lam)
                ret=adv+vals
                var_y=torch.var(ret)
                ev=float(1.0 - torch.var(ret-vals)/var_y) if var_y>0 else 0.0
            except Exception:
                ev=0.0
            std=float(np.std(self.buffer.act[:self.buffer.ptr]))
        else:
            ev=0.0
            std=0.0
        m={
            "rollout": {"ep_len_mean": int(np.mean(self.ep_length_history[-50:])) if self.ep_length_history else 0, "ep_rew_mean": float(np.mean(self.reward_history[-self.update_interval:])) if self.reward_history else 0.0},
            "time": {"fps": int((self.step_count)/(max(1,time.time()-self.start_time))), "iterations": self.iterations, "time_elapsed": int(time.time()-self.start_time), "total_timesteps": self.step_count},
            "train": {"approx_kl": float(0.0), "clip_fraction": 0, "clip_range": self.clip, "entropy_loss": float(-std), "explained_variance": float(ev), "learning_rate": self.lr, "loss": float(0.0), "n_updates": self.n_updates, "policy_gradient_loss": float(0.0), "std": float(std), "value_loss": float(0.0)}
        }
        return m
    def _print_box(self,m):
        print("----------------------------------------------")
        for section in ["rollout","time","train"]:
            sname=section+"/"
            left="| "+sname+" "*(28-len(sname))
            print(left+"|"+" "*13+"|")
            for k,v in m[section].items():
                name=f"| | {k}"
                pad=28-len(k)-3
                val=str(v)
                print(name+" "*pad+"| "+val+" "*(13-len(val))+"|")
        print("----------------------------------------------")
