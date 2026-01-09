import os
import json
import zipfile
from pathlib import Path
import sys
import numpy as np
import torch
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
# ensure project root is importable for updater
try:
    ROOT=Path(__file__).resolve().parents[2]
    p=str(ROOT)
    if p not in sys.path:
        sys.path.insert(0,p)
except Exception:
    pass
try:
    import updater
except Exception:
    updater=None

class TekksTrainerBot(BaseAgent):
    def initialize_agent(self):
        try:
            if updater is not None:
                updated = updater.check_for_update_blocking_ui(__file__)
                if updated:
                    raise SystemExit("TekksTrainer updated. Please restart RLBot.")
        except Exception:
            pass
        self.model=None
        self._logged_missing_model=False
        self._logged_loaded=False
        self._logged_mismatch=False
        self._obs_dim_loaded=None
        # make sure a model exists by copying from training outputs if present
        try:
            root=Path(__file__).resolve().parents[2]
            models_dir=Path(__file__).parent.joinpath("models")
            models_dir.mkdir(parents=True, exist_ok=True)
            target=models_dir.joinpath("latest_ppo.zip")
            def _pick_latest():
                final=root.joinpath("outputs","final","final_ppo.zip")
                if final.exists():
                    return final
                ck_dir=root.joinpath("outputs","checkpoints")
                if ck_dir.exists():
                    cks=sorted([p for p in ck_dir.iterdir() if p.name.endswith("_ppo.zip")])
                    if cks:
                        return cks[-1]
                return None
            if not target.exists():
                src=_pick_latest()
                if src is not None:
                    import shutil
                    shutil.copy2(str(src), str(target))
                    try:
                        self.logger.info(f"TekksTrainer: copied model from {src} to {target}")
                    except Exception:
                        print(f"TekksTrainer: copied model from {src} to {target}")
        except Exception:
            pass
        try:
            mp=Path(__file__).parent.joinpath("models").joinpath("latest_ppo.zip")
            if mp.exists():
                import io
                with zipfile.ZipFile(str(mp),"r") as z:
                    meta_json=z.read("meta.json") if "meta.json" in z.namelist() else b"{}"
                    meta=json.loads(meta_json.decode("utf-8"))
                    obs_dim=int(meta.get("obs_dim",63))
                    self._obs_dim_loaded=obs_dim
                    if "model.pt" not in z.namelist():
                        try:
                            self.logger.info("TekksTrainer: latest_ppo.zip missing model.pt")
                        except Exception:
                            print("TekksTrainer: latest_ppo.zip missing model.pt")
                        self.model=None
                        return
                    buf=z.read("model.pt")
                    bio=io.BytesIO(buf)
                    state=torch.load(bio,map_location="cpu")
                self.model=torch.jit.script(torch.nn.Sequential(torch.nn.Linear(obs_dim,256),torch.nn.Tanh(),torch.nn.Linear(256,256),torch.nn.Tanh(),torch.nn.Linear(256,7)))
                try:
                    self.model.load_state_dict(state,strict=False)
                    try:
                        self.logger.info(f"TekksTrainer: model loaded (obs_dim={obs_dim})")
                    except Exception:
                        print(f"TekksTrainer: model loaded (obs_dim={obs_dim})")
                    self._logged_loaded=True
                except Exception:
                    try:
                        self.logger.info("TekksTrainer: failed to load state_dict")
                    except Exception:
                        print("TekksTrainer: failed to load state_dict")
                    self.model=None
            
        except Exception:
            self.model=None
            if not self._logged_missing_model:
                try:
                    if not Path(__file__).parent.joinpath("models").joinpath("latest_ppo.zip").exists():
                        self.logger.info("TekksTrainer: no model found at rlbot/TekksTrainer/models/latest_ppo.zip")
                    else:
                        self.logger.info("TekksTrainer: exception during model load")
                except Exception:
                    if not Path(__file__).parent.joinpath("models").joinpath("latest_ppo.zip").exists():
                        print("TekksTrainer: no model found at rlbot/TekksTrainer/models/latest_ppo.zip")
                    else:
                        print("TekksTrainer: exception during model load")
                self._logged_missing_model=True
    def get_output(self,packet):
        me=packet.game_cars[self.index]
        opp_idx=0 if self.index!=0 else 1
        opp=packet.game_cars[opp_idx]
        ball=packet.game_ball.physics
        own_goal_x=-5120 if me.team==0 else 5120
        opp_goal_x=5120 if me.team==0 else -5120
        p=me.physics.location
        v=me.physics.velocity
        av=me.physics.angular_velocity
        pitch=me.physics.rotation.pitch
        yaw=me.physics.rotation.yaw
        roll=me.physics.rotation.roll
        def fwd():
            cp, cy, cr = pitch, yaw, roll
            x = np.cos(cp)*np.cos(cy)
            y = np.cos(cp)*np.sin(cy)
            z = np.sin(cp)
            return x,y,z
        fx,fy,fz=fwd()
        rx,ry,rz = -np.sin(yaw), np.cos(yaw), 0.0
        ux,uy,uz = -np.sin(pitch)*np.cos(yaw), -np.sin(pitch)*np.sin(yaw), np.cos(pitch)
        b=ball.location
        bv=ball.velocity
        bspd=float(np.sqrt(bv.x*bv.x+bv.y*bv.y+bv.z*bv.z))
        cbx=b.x-p.x; cby=b.y-p.y; cbz=b.z-p.z
        cbl=float(np.sqrt(cbx*cbx+cby*cby+cbz*cbz))+1e-6
        cbn=(cbx/cbl,cby/cbl,cbz/cbl)
        dist_ball=cbl
        dist_own=float(np.sqrt((p.x-own_goal_x)**2+p.y**2+p.z**2))
        dist_opp=float(np.sqrt((p.x-opp_goal_x)**2+p.y**2+p.z**2))
        op=opp.physics.location
        ov=opp.physics.velocity
        ofx,ofy,ofz = fx,fy,fz
        navx,navy,navz,nav_a = 0.0,0.0,0.0,0.0
        nbx,nby,nbz,nb_a = 0.0,0.0,0.0,0.0
        step_left=0.0
        status=[
            float(getattr(me,'boost',0.0)),
            1.0 if getattr(me,'has_wheel_contact',False) else 0.0,
            1.0 if (getattr(me,'is_jumping',False) or getattr(me,'jumped',False)) else 0.0,
            1.0 if (getattr(me,'has_double_jump',False) or getattr(me,'double_jumped',False)) else 0.0,
            1.0 if getattr(me,'is_super_sonic',False) else 0.0
        ]
        obs=np.array([
            p.x,p.y,p.z,v.x,v.y,v.z,av.x,av.y,av.z,
            fx,fy,fz,rx,ry,rz,ux,uy,uz,
            pitch,yaw,roll]+status+[
            b.x,b.y,b.z,bv.x,bv.y,bv.z,bspd,cbn[0],cbn[1],cbn[2],dist_ball,
            own_goal_x,0.0,0.0,opp_goal_x,0.0,0.0,dist_own,dist_opp,
            op.x,op.y,op.z,ov.x,ov.y,ov.z,ofx,ofy,ofz,
            navx,navy,navz,nav_a,nbx,nby,nbz,nb_a,step_left
        ],dtype=np.float32)
        if self.model is not None:
            with torch.no_grad():
                x=torch.tensor(obs).unsqueeze(0)
                if self._obs_dim_loaded is not None and x.shape[1]!=self._obs_dim_loaded:
                    # adapt by truncating or padding zeros
                    if not self._logged_mismatch:
                        try:
                            self.logger.info(f"TekksTrainer: obs_dim mismatch (got {x.shape[1]}, expected {self._obs_dim_loaded}); adapting")
                        except Exception:
                            print(f"TekksTrainer: obs_dim mismatch (got {x.shape[1]}, expected {self._obs_dim_loaded}); adapting")
                        self._logged_mismatch=True
                    if x.shape[1]>self._obs_dim_loaded:
                        x=x[:,:self._obs_dim_loaded]
                    else:
                        pad=torch.zeros((1,self._obs_dim_loaded-x.shape[1]),dtype=x.dtype)
                        x=torch.cat([x,pad],dim=1)
                a=self.model(x).squeeze(0).numpy()
        else:
            if not self._logged_missing_model:
                try:
                    self.logger.info("TekksTrainer: model not loaded; outputs are neutral controls")
                except Exception:
                    print("TekksTrainer: model not loaded; outputs are neutral controls")
                self._logged_missing_model=True
            a=np.zeros(7,dtype=np.float32)
        ctrl=SimpleControllerState()
        ctrl.throttle=float(np.clip(a[0],-1.0,1.0))
        ctrl.jump=bool(a[1]>0.5)
        ctrl.boost=bool(a[2]>0.5)
        ctrl.steer=float(np.clip(a[3],-1.0,1.0))
        ctrl.pitch=float(np.clip(a[4],-1.0,1.0))
        roll=float(np.clip(a[6]-a[5],-1.0,1.0))
        ctrl.roll=roll
        return ctrl
