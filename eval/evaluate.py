import os
import json
import time
import sys
from pathlib import Path
import numpy as np
import torch
from env.rocketsim_env import RocketSim1v1Env
import updater

def load_policy(zip_path):
    try:
        import zipfile, io
        with zipfile.ZipFile(zip_path,"r") as z:
            buf=z.read("model.pt")
            bio=io.BytesIO(buf)
            state=torch.load(bio,map_location="cpu")
        return state
    except Exception:
        return None

def evaluate_model(model_zip,cfg):
    env=RocketSim1v1Env(cfg)
    o_blue,_=env.reset()
    obs_dim=len(o_blue)
    act_dim=7
    net=torch.jit.script(torch.nn.Sequential(torch.nn.Linear(obs_dim,256),torch.nn.Tanh(),torch.nn.Linear(256,256),torch.nn.Tanh(),torch.nn.Linear(256,act_dim)))
    state=load_policy(model_zip)
    if state is not None:
        try:
            net.load_state_dict(state,strict=False)
        except Exception:
            pass
    scenarios=[{"name":"standard","ball":[0,0,93]},{"name":"corner","ball":[-400,300,93]},{"name":"mid_air","ball":[0,0,300]}]
    strategies=["chase","goalie","boost"]
    metrics={"mean_reward":0.0,"ball_touches":0,"episodes":0,"mean_episode_length":0.0,"strategies":{}}
    for st in strategies:
        metrics["strategies"][st]={"mean_reward":0.0,"episodes":0,"touches":0,"goals_for":0,"goals_against":0}
    for sc in scenarios:
        for st in strategies:
            o_blue,_=env.reset()
            ep_r=0.0
            touches=0
            done=False
            steps=0
            while not done and steps<env.cfg["env"]["tick_rate"]*30:
                obs=torch.tensor(o_blue,dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    a=net(obs).squeeze(0).numpy()
                o_b,o_o,r_b,r_o,done=env.step(a,heuristic_action(env,st))
                o_blue=o_b
                ep_r+=r_b
                if env._car_touched_ball(env.blue):
                    touches+=1
                bx=env.ball.position.x
                if bx>5200:
                    metrics["strategies"][st]["goals_for"]+=1
                    done=True
                if bx<-5200:
                    metrics["strategies"][st]["goals_against"]+=1
                    done=True
                steps+=1
            metrics["mean_reward"]+=ep_r
            metrics["ball_touches"]+=touches
            metrics["mean_episode_length"]+=steps
            metrics["episodes"]+=1
            s=metrics["strategies"][st]
            s["mean_reward"]+=ep_r
            s["episodes"]+=1
            s["touches"]+=touches
    if metrics["episodes"]>0:
        metrics["mean_reward"]=metrics["mean_reward"]/metrics["episodes"]
        metrics["mean_episode_length"]=metrics["mean_episode_length"]/metrics["episodes"]
    for st in strategies:
        s=metrics["strategies"][st]
        if s["episodes"]>0:
            s["mean_reward"]=s["mean_reward"]/s["episodes"]
    return metrics

def heuristic_action(env,kind):
    blue=env.blue
    orange=env.orange
    ball=env.ball
    if kind=="chase":
        steer=1.0 if ball.position.y>orange.position.y else -1.0
        return np.array([0.8,0.0,1.0,steer,0.0,0.0,0.0],dtype=np.float32)
    if kind=="goalie":
        steer=-1.0 if ball.position.x<0 else 1.0
        return np.array([0.6,0.0,0.0,steer,0.0,0.0,0.0],dtype=np.float32)
    if kind=="boost":
        return np.array([1.0,0.0,1.0,0.0,0.0,0.0,0.0],dtype=np.float32)
    return np.zeros(7,dtype=np.float32)
    if metrics["episodes"]>0:
        metrics["mean_reward"]=metrics["mean_reward"]/metrics["episodes"]
    return metrics

def main():
    updater.check_for_update(__file__)
    if not (hasattr(sys, "base_prefix") and sys.prefix!=sys.base_prefix) and not os.getenv("VIRTUAL_ENV"):
        raise RuntimeError("venv required. Activate a virtual environment before running.")
    with open("config.json","r") as f:
        cfg=json.load(f)
    out=cfg["paths"]["eval"]
    Path(out).mkdir(parents=True,exist_ok=True)
    ckdir=cfg["paths"]["final"]
    files=[f for f in os.listdir(ckdir) if f.endswith("_ppo.zip") or f.endswith("final_ppo.zip")]
    results={}
    for f in files:
        p=os.path.join(ckdir,f)
        res=evaluate_model(p,cfg)
        results[f]=res
    with open(os.path.join(out,"results.json"),"w") as g:
        json.dump(results,g,indent=2)

if __name__=="__main__":
    main()
