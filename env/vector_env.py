import os
import sys
import time
import multiprocessing as mp
import numpy as np
import copy
from env.rocketsim_env import RocketSim1v1Env

def _worker(conn, cfg):
    try:
        env=RocketSim1v1Env(cfg)
        ob_blue, ob_orange = env.reset()
        conn.send((ob_blue, ob_orange))
        while True:
            msg=conn.recv()
            if msg[0]=="step":
                a_blue=msg[1]
                ob_blue, ob_orange, r_b, r_o, done=env.self_play_step(a_blue)
                if done:
                    ob_blue, ob_orange=env.reset()
                conn.send((ob_blue, ob_orange, r_b, r_o, done))
            elif msg[0]=="reset":
                ob_blue, ob_orange=env.reset()
                conn.send((ob_blue, ob_orange))
            elif msg[0]=="close":
                break
    except Exception as e:
        try:
            conn.send(("error", str(e)))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

class SubprocVecEnv:
    def __init__(self, cfg, n_envs):
        self.n=n_envs
        self.parents=[]
        self.children=[]
        self.procs=[]
        for i in range(n_envs):
            parent, child=mp.Pipe()
            cfg_i=copy.deepcopy(cfg)
            cfg_i.setdefault("env",{})
            cfg_i["env"]["write_match_state"]= (i==0)
            cfg_i["env"]["state_write_interval"]= cfg_i["env"].get("state_write_interval",0.05)
            p=mp.Process(target=_worker,args=(child,cfg_i))
            p.daemon=True
            p.start()
            self.parents.append(parent)
            self.children.append(child)
            self.procs.append(p)
        obs=[]
        for pr in self.parents:
            o=pr.recv()
            obs.append(o[0])
        self.obs=np.stack(obs,axis=0)
    def step(self, actions):
        for pr,a in zip(self.parents,actions):
            pr.send(("step", a))
        results=[pr.recv() for pr in self.parents]
        ob_blue=[r[0] for r in results]
        ob_orange=[r[1] for r in results]
        r_b=[r[2] for r in results]
        r_o=[r[3] for r in results]
        done=[r[4] for r in results]
        return np.stack(ob_blue), np.stack(ob_orange), np.array(r_b,dtype=np.float32), np.array(r_o,dtype=np.float32), np.array(done,dtype=np.bool_) 
    def reset(self):
        for pr in self.parents:
            pr.send(("reset", None))
        obs=[pr.recv() for pr in self.parents]
        ob_blue=[o[0] for o in obs]
        ob_orange=[o[1] for o in obs]
        return np.stack(ob_blue), np.stack(ob_orange)
    def close(self):
        for pr in self.parents:
            try:
                pr.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            try:
                p.join(timeout=1.0)
            except Exception:
                pass
