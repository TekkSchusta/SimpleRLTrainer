import os
import json
import math
import random
from pathlib import Path
import numpy as np
try:
    import rocketsim as rs
except Exception:
    try:
        import RocketSim as rs
    except Exception:
        rs=None

class SimplePolicy:
    def __init__(self, zip_path=None, obs_dim=30, act_dim=7):
        self.valid=False
        self.obs_dim=obs_dim
        self.act_dim=act_dim
        try:
            import torch, zipfile, io
            self.net=torch.jit.script(torch.nn.Sequential(torch.nn.Linear(obs_dim,256),torch.nn.Tanh(),torch.nn.Linear(256,256),torch.nn.Tanh(),torch.nn.Linear(256,act_dim)))
            if zip_path and Path(zip_path).exists():
                with zipfile.ZipFile(zip_path,"r") as z:
                    buf=z.read("model.pt")
                    bio=io.BytesIO(buf)
                    state=torch.load(bio,map_location="cpu")
                try:
                    self.net.load_state_dict(state,strict=False)
                except Exception:
                    pass
                self.net.eval()
                self.valid=True
            else:
                self.valid=False
        except Exception:
            self.valid=False
    def act(self,obs):
        if self.valid:
            import torch
            with torch.no_grad():
                a=self.net(torch.tensor(obs,dtype=torch.float32).unsqueeze(0))
                return a.squeeze(0).numpy()
        return np.zeros(self.act_dim,dtype=np.float32)

class RocketSim1v1Env:
    def __init__(self,cfg):
        if rs is None:
            raise RuntimeError("rocketsim not available")
        self.cfg=cfg
        self.dt=1.0/float(cfg["env"]["tick_rate"]) if cfg and "env" in cfg else 1.0/120.0
        self.max_seconds=cfg["env"]["max_episode_seconds"]
        self.ball_touch_reward=cfg["env"]["ball_touch_reward"]
        self.backwards_penalty=cfg["env"]["backwards_penalty"]
        self.upside_down_penalty=cfg["env"]["upside_down_penalty"]
        self.paths=cfg["paths"]
        self.write_match_state=bool(cfg.get("env",{}).get("write_match_state", True))
        self.state_write_interval=float(cfg.get("env",{}).get("state_write_interval", 0.05))
        self._last_state_write=0.0
        self.boost_use_penalty=float(cfg.get("env",{}).get("boost_use_penalty", -0.001))
        self.dist_increase_penalty=float(cfg.get("env",{}).get("dist_increase_penalty", -0.0002))
        self.air_time_penalty=float(cfg.get("env",{}).get("air_time_penalty", -0.0005))
        mesh_path=os.getenv("RS_COLLISION_MESHES", "collision_meshes")
        rs.init(mesh_path)
        self.arena=rs.Arena(rs.GameMode.SOCCAR)
        self.blue=self.arena.add_car(rs.Team.BLUE, rs.CarConfig())
        self.orange=self.arena.add_car(rs.Team.ORANGE, rs.CarConfig())
        self.ball=self.arena.ball
        self.episode_time=0.0
        self.last_ball_touch_blue=False
        self.last_ball_touch_orange=False
        self.opponent_policy=None
        self._refresh_opponent_policy()
        self._write_match_state()
        self._init_shaping_trackers()
    def reset(self):
        self.arena.reset_kickoff()
        self.episode_time=0.0
        self.last_ball_touch_blue=False
        self.last_ball_touch_orange=False
        self._init_shaping_trackers()
        self._write_match_state()
        return self._get_obs(0), self._get_obs(1)
    def step(self,action_blue,action_orange):
        bc=self._to_controls(action_blue)
        oc=self._to_controls(action_orange)
        self.blue.set_controls(bc)
        self.orange.set_controls(oc)
        self.arena.step(1)
        self.episode_time+=1.0/float(self.arena.tick_rate)
        r_blue=self._compute_reward(0)
        r_orange=self._compute_reward(1)
        done=self.episode_time>=self.max_seconds
        self._write_match_state()
        return self._get_obs(0), self._get_obs(1), r_blue, r_orange, done
    def self_play_step(self,action_blue):
        if self.opponent_policy is None:
            self._refresh_opponent_policy()
        obs_orange=self._get_obs(1)
        action_orange=self.opponent_policy.act(obs_orange)
        return self.step(action_blue,action_orange)
    def _init_shaping_trackers(self):
        try:
            bs=self.ball.get_state()
            bpos=bs.pos
            sblue=self.blue.get_state()
            sorange=self.orange.get_state()
            self._last_boost_blue=float(getattr(sblue,'boost',0.0))
            self._last_boost_orange=float(getattr(sorange,'boost',0.0))
            self._last_dist_blue=self._dist(sblue.pos,bpos)
            self._last_dist_orange=self._dist(sorange.pos,bpos)
        except Exception:
            self._last_boost_blue=0.0
            self._last_boost_orange=0.0
            self._last_dist_blue=0.0
            self._last_dist_orange=0.0
    def _get_obs(self,idx):
        car=self.blue if idx==0 else self.orange
        opp=self.orange if idx==0 else self.blue
        cs=car.get_state()
        p=cs.pos
        v=cs.vel
        av=cs.ang_vel if hasattr(cs,'ang_vel') else rs.Vec(0,0,0)
        fd=car.get_forward_dir()
        rd=car.get_right_dir()
        ud=car.get_up_dir()
        rmat=cs.rot_mat
        yaw,pitch,roll=self._rotmat_to_euler(rmat)
        bs=self.ball.get_state()
        bpos=bs.pos
        bvel=bs.vel
        bspd=math.sqrt(bvel.x*bvel.x+bvel.y*bvel.y+bvel.z*bvel.z)
        cbx=bpos.x-p.x
        cby=bpos.y-p.y
        cbz=bpos.z-p.z
        cbl=math.sqrt(cbx*cbx+cby*cby+cbz*cbz)+1e-6
        cbn=(cbx/cbl,cby/cbl,cbz/cbl)
        pads=self.arena.get_boost_pads()
        nearest_av=None
        nearest_big=None
        for pad in pads:
            ps=pad.get_state()
            d=self._dist(pad.get_pos(),p)
            if ps.is_active and (nearest_av is None or d<nearest_av[0]):
                nearest_av=(d,pad.get_pos(),1.0)
            if pad.is_big and (nearest_big is None or d<nearest_big[0]):
                nearest_big=(d,pad.get_pos(),1.0 if ps.is_active else 0.0)
        nav=nearest_av[1] if nearest_av else rs.Vec(0,0,0)
        nav_active=nearest_av[2] if nearest_av else 0.0
        nbig=nearest_big[1] if nearest_big else rs.Vec(0,0,0)
        nbig_active=nearest_big[2] if nearest_big else 0.0
        own_goal=rs.Vec(-5120,0,0) if (idx==0) else rs.Vec(5120,0,0)
        opp_goal=rs.Vec(5120,0,0) if (idx==0) else rs.Vec(-5120,0,0)
        os=opp.get_state()
        o=os.pos
        ovel=os.vel
        ofd=opp.get_forward_dir()
        dist_ball=self._dist(p,bpos)
        dist_own=self._dist(p,own_goal)
        dist_opp=self._dist(p,opp_goal)
        step_left=max(0.0,self.max_seconds-self.episode_time)
        status=[float(getattr(cs,'boost',0.0)), 1.0 if getattr(cs,'is_on_ground',False) else 0.0, 1.0 if getattr(cs,'is_jumping',False) else 0.0, 1.0 if getattr(cs,'has_flip_or_jump',False) else 0.0, 1.0 if getattr(cs,'is_supersonic',False) else 0.0]
        arr=[p.x,p.y,p.z,v.x,v.y,v.z,av.x,av.y,av.z,fd.x,fd.y,fd.z,rd.x,rd.y,rd.z,ud.x,ud.y,ud.z,pitch,yaw,roll]+status+[bpos.x,bpos.y,bpos.z,bvel.x,bvel.y,bvel.z,bspd,cbn[0],cbn[1],cbn[2],dist_ball,own_goal.x,own_goal.y,own_goal.z,opp_goal.x,opp_goal.y,opp_goal.z,dist_own,dist_opp,o.x,o.y,o.z,ovel.x,ovel.y,ovel.z,ofd.x,ofd.y,ofd.z,nav.x,nav.y,nav.z,nav_active,nbig.x,nbig.y,nbig.z,nbig_active,step_left]
        return np.array(arr,dtype=np.float32)
    def _compute_reward(self,idx):
        car=self.blue if idx==0 else self.orange
        self_touch=self._car_touched_ball(car)
        r=0.0
        if self_touch:
            r+=self.ball_touch_reward
        cs=car.get_state()
        fd=car.get_forward_dir()
        fv=fd.x*cs.vel.x+fd.y*cs.vel.y+fd.z*cs.vel.z
        if fv<0.0:
            r+=self.backwards_penalty
        rmat=cs.rot_mat
        _,_,roll=self._rotmat_to_euler(rmat)
        if abs(roll)>2.7:
            r+=self.upside_down_penalty
        try:
            bs=self.ball.get_state()
            d=self._dist(cs.pos,bs.pos)
            tbx=bs.pos.x-cs.pos.x
            tby=bs.pos.y-cs.pos.y
            tbz=bs.pos.z-cs.pos.z
            tn=max(1e-6, math.sqrt(tbx*tbx+tby*tby+tbz*tbz))
            tbu=(tbx/tn,tby/tn,tbz/tn)
            ogx=-5120 if idx==0 else 5120
            og=(ogx,0.0,0.0)
            gox=5120 if idx==0 else -5120
            gg=(gox,0.0,0.0)
            togx=ogx-cs.pos.x
            togy=0.0-cs.pos.y
            togz=0.0-cs.pos.z
            gon=max(1e-6, math.sqrt(togx*togx+togy*togy+togz*togz))
            gou=(togx/gon,togy/gon,togz/gon)
            fdu=(fd.x,fd.y,fd.z)
            fdn=max(1e-6, math.sqrt(fd.x*fd.x+fd.y*fd.y+fd.z*fd.z))
            fdu=(fdu[0]/fdn,fdu[1]/fdn,fdu[2]/fdn)
            dot_ball=fdu[0]*tbu[0]+fdu[1]*tbu[1]+fdu[2]*tbu[2]
            dot_goal=fdu[0]*gou[0]+fdu[1]*gou[1]+fdu[2]*gou[2]
            attack=(dot_ball>0.6 and d<1600)
            dbgx=bs.pos.x-ogx
            dbgy=bs.pos.y-0.0
            dbgz=bs.pos.z-0.0
            dist_ball_goal=math.sqrt(dbgx*dbgx+dbgy*dbgy+dbgz*dbgz)
            defend=(dist_ball_goal<2200) or (dot_goal>0.5)
            reposition=(dot_ball>0.4 and d>1800)
            if idx==0:
                boost_prev=self._last_boost_blue
                dist_prev=self._last_dist_blue
                self._last_boost_blue=float(getattr(cs,'boost',boost_prev))
                self._last_dist_blue=d
            else:
                boost_prev=self._last_boost_orange
                dist_prev=self._last_dist_orange
                self._last_boost_orange=float(getattr(cs,'boost',boost_prev))
                self._last_dist_orange=d
            boost_now=float(getattr(cs,'boost',boost_prev))
            boost_drop=max(0.0, boost_prev - boost_now)
            intent_mult=0.25 if (attack or defend or reposition) else 2.0
            r+=self.boost_use_penalty*intent_mult*boost_drop
            dist_delta=d - dist_prev
            if dist_delta>0.0:
                r+=self.dist_increase_penalty*dist_delta
            if not cs.is_on_ground:
                air_mult=0.3 if (attack or defend) else 2.0
                r+=self.air_time_penalty*air_mult
        except Exception:
            pass
        return r
    def _car_touched_ball(self,car):
        cs=car.get_state()
        cp=cs.pos
        bs=self.ball.get_state()
        bp=bs.pos
        return self._dist(cp,bp)<120.0
    def _to_controls(self,a):
        throttle=float(np.clip(a[0],-1.0,1.0))
        jump=bool(a[1]>0.5)
        boost=bool(a[2]>0.5)
        yaw=float(np.clip(a[3],-1.0,1.0))
        pitch=float(np.clip(a[4],-1.0,1.0))
        arl=float(np.clip(a[5],0.0,1.0))
        arr=float(np.clip(a[6],0.0,1.0))
        roll=float(arr-arl)
        return rs.CarControls(throttle=throttle, steer=yaw, pitch=pitch, yaw=yaw, roll=roll, boost=boost, handbrake=False, jump=jump)
    def _dist(self,a,b):
        dx=a.x-b.x
        dy=a.y-b.y
        dz=a.z-b.z
        return math.sqrt(dx*dx+dy*dy+dz*dz)
    def _refresh_opponent_policy(self):
        ckdir=self.paths["checkpoints"]
        Path(ckdir).mkdir(parents=True,exist_ok=True)
        files=[f for f in os.listdir(ckdir) if f.endswith("_ppo.zip")]
        if not files:
            self.opponent_policy=SimplePolicy(None)
            return
        files_sorted=sorted(files)
        newest=files_sorted[-1]
        steps=[self._extract_step(x) for x in files_sorted]
        newest_step=steps[-1]
        cutoff=int(newest_step*0.8)
        candidate=None
        for f,s in zip(files_sorted,steps):
            if s<=cutoff:
                candidate=f
        if candidate is None:
            candidate=files_sorted[0]
        mp=os.path.join(ckdir,candidate)
        self.opponent_policy=SimplePolicy(mp,obs_dim=len(self._get_obs(0)),act_dim=7)
    def _rotmat_to_euler(self,rm):
        r11,r12,r13=rm[0]
        r21,r22,r23=rm[1]
        r31,r32,r33=rm[2]
        sy=math.sqrt(r11*r11+r21*r21)
        if sy>1e-6:
            yaw=math.atan2(r21,r11)
            pitch=math.atan2(-r31,sy)
            roll=math.atan2(r32,r33)
        else:
            yaw=math.atan2(-r12,r22)
            pitch=math.atan2(-r31,sy)
            roll=0.0
        return yaw,pitch,roll
    def _extract_step(self,name):
        try:
            parts=name.split("_")
            for p in parts:
                if p.isdigit():
                    return int(p)
        except Exception:
            return 0
        return 0
    def _write_match_state(self):
        try:
            import time as _t
            if not self.write_match_state:
                return
            now=_t.time()
            if now-self._last_state_write<self.state_write_interval:
                return
            bs=self.ball.get_state()
            bpos=[bs.pos.x,bs.pos.y,bs.pos.z]
            csb=self.blue.get_state()
            cob=self.orange.get_state()
            bcar=[csb.pos.x,csb.pos.y,csb.pos.z]
            ocar=[cob.pos.x,cob.pos.y,cob.pos.z]
            bdir=self.blue.get_forward_dir()
            odir=self.orange.get_forward_dir()
            pads=[]
            for p in self.arena.get_boost_pads():
                s=p.get_state()
                pv=p.get_pos()
                pads.append({"pos":[pv.x,pv.y,pv.z],"is_big":bool(p.is_big),"is_active":bool(s.is_active)})
            state={
                "blue":{"pos":bcar,"fwd":[bdir.x,bdir.y,bdir.z]},
                "orange":{"pos":ocar,"fwd":[odir.x,odir.y,odir.z]},
                "ball":{"pos":bpos},
                "boost_pads":pads,
                "time":self.episode_time
            }
            with open(self.paths["live_match_state"],"w") as f:
                json.dump(state,f)
            self._last_state_write=now
        except Exception:
            pass
