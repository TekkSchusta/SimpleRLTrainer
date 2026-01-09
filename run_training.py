import json
import os
import threading
import time
import sys
import argparse
from pathlib import Path
from trainer.ppo_trainer import PPOTrainer
from env.rocketsim_env import RocketSim1v1Env
from env.vector_env import SubprocVecEnv
import updater

def ensure_dirs(cfg):
    for k in ["outputs","checkpoints","final","eval","backups"]:
        Path(cfg["paths"][k]).mkdir(parents=True, exist_ok=True)

def load_config():
    with open("config.json","r") as f:
        return json.load(f)

def _make_session(cfg, args):
    base_out = cfg["paths"]["outputs"]
    sessions_dir = os.path.join(base_out, "sessions")
    Path(sessions_dir).mkdir(parents=True, exist_ok=True)
    if args and getattr(args, "session", None):
        sid = args.session
        session_dir = os.path.join(sessions_dir, sid)
    else:
        sid = f"{int(time.time())}_{os.getpid()}"
        session_dir = os.path.join(sessions_dir, sid)
    Path(session_dir).mkdir(parents=True, exist_ok=True)
    cfg["paths"]["session_dir"] = session_dir
    cfg["paths"]["live_metrics"] = os.path.join(session_dir, "live_metrics.json")
    cfg["paths"]["live_match_state"] = os.path.join(session_dir, "live_match_state.json")
    return sid

def start_training(args=None):
    updater.check_for_update(__file__)
    if not (hasattr(sys, "base_prefix") and sys.prefix!=sys.base_prefix) and not os.getenv("VIRTUAL_ENV"):
        raise RuntimeError("venv required. Activate a virtual environment before running.")
    cfg=load_config()
    if args:
        if args.total_steps is not None:
            cfg["training"]["total_steps"]=int(args.total_steps)
        if args.ck_interval is not None:
            cfg["training"]["checkpoint_interval"]=int(args.ck_interval)
        if args.num_envs is not None and int(args.num_envs)>1:
            cfg["training"]["num_envs"]=int(args.num_envs)
    ensure_dirs(cfg)
    sid=_make_session(cfg, args)
    n_envs=int(cfg["training"].get("num_envs", max(1,(os.cpu_count() or 2)//2)))
    env=SubprocVecEnv(cfg,n_envs) if n_envs>1 else RocketSim1v1Env(cfg)
    trainer=PPOTrainer(env,cfg)
    print(f"TekksTrainer training start (session {sid})")
    trainer.train()
    print("TekksTrainer training complete")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--total_steps",type=int)
    p.add_argument("--ck_interval",type=int)
    p.add_argument("--num_envs",type=int)
    p.add_argument("--session",type=str,help="Attach to a named session id")
    p.add_argument("--resume",type=str,choices=["auto","final","checkpoint"],help="Resume from previous model")
    args=p.parse_args()
    try:
        if args.resume:
            cfg=load_config()
            cfg.setdefault("training",{})
            cfg["training"]["resume_mode"]=args.resume
            with open("config.json","w") as f:
                json.dump(cfg,f,indent=2)
        start_training(args)
    except Exception as e:
        print(str(e))
