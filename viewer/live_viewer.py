import json
import time
import threading
import math
import os
import sys
from pathlib import Path
import tkinter as tk
import updater

class LiveViewer:
    def __init__(self,metrics_path,match_path):
        self.metrics_path=metrics_path
        self.match_path=match_path
        self.root=tk.Tk()
        self.root.title("TekksTrainer Viewer")
        self.root.configure(bg="#0e0e10")
        self.canvas=tk.Canvas(self.root,width=900,height=560,bg="#121217",highlightthickness=0)
        self.canvas.grid(row=0,column=0,sticky="nsew")
        self.side=tk.Frame(self.root,bg="#0e0e10")
        self.side.grid(row=0,column=1,sticky="ns")
        self.root.grid_columnconfigure(0,weight=1)
        self.root.grid_rowconfigure(0,weight=1)
        self.info=tk.Label(self.side,text="",fg="#eaeaf0",bg="#0e0e10")
        self.info.pack(anchor="nw",padx=12,pady=8)
        self.running=True
        self.last_read=0.0
        self.field_pad=40
        self.world_x=5120
        self.world_y=4096
        # vertical orientation: x (field length) maps to screen Y; y maps to screen X
        self.scale_x=(900-2*self.field_pad)/(self.world_y*2)
        self.scale_y=(560-2*self.field_pad)/(self.world_x*2)
        # persistent items to avoid flicker
        self.items={"bg":[],"ball":None,"blue":None,"orange":None,"blue_dir":None,"orange_dir":None,"pads":[]}
        # draw static background once
        self._draw_background()
        self.canvas.bind_all("<space>",self._toggle_pause)
        self.paused=False
        self.last_metrics_mtime=0.0
        self.last_match_mtime=0.0
        self.prev_state=None
        self.curr_state=None
        threading.Thread(target=self.loop,daemon=True).start()
        self.root.protocol("WM_DELETE_WINDOW",self.stop)
    def _toggle_pause(self,_):
        self.paused=not self.paused
    def stop(self):
        self.running=False
        self.root.destroy()
    def _map(self,x,y):
        # vertical orientation: X -> screen Y, Y -> screen X
        sx=(y+self.world_y)*self.scale_x+self.field_pad
        sy=(x+self.world_x)*self.scale_y+self.field_pad
        return int(sx), int(560-int(sy))
    def _draw_background(self):
        # field rectangle
        self.items["bg"].append(self.canvas.create_rectangle(self.field_pad,self.field_pad,900-self.field_pad,560-self.field_pad,outline="#2aa74e"))
        # mid line (vertical orientation): through center X (screen)
        mid_x=(900)//2
        self.items["bg"].append(self.canvas.create_line(mid_x,self.field_pad,mid_x,560-self.field_pad,fill="#2aa74e"))
    def draw(self,state,metrics):
        # ball
        bx,by,_=state["ball"]["pos"]
        sbx,sby=self._map(bx,by)
        if self.items["ball"] is None:
            self.items["ball"]=self.canvas.create_oval(sbx-7,sby-7,sbx+7,sby+7,fill="#f0f0f0",outline="")
        else:
            self.canvas.coords(self.items["ball"],sbx-7,sby-7,sbx+7,sby+7)
        # cars
        bcx,bcy,_=state["blue"]["pos"]
        ocx,ocy,_=state["orange"]["pos"]
        sbcx,sbcy=self._map(bcx,bcy)
        socx,socy=self._map(ocx,ocy)
        if self.items["blue"] is None:
            self.items["blue"]=self.canvas.create_rectangle(sbcx-10,sbcy-10,sbcx+10,sbcy+10,fill="#3da7ff",outline="")
        else:
            self.canvas.coords(self.items["blue"],sbcx-10,sbcy-10,sbcx+10,sbcy+10)
        if self.items["orange"] is None:
            self.items["orange"]=self.canvas.create_rectangle(socx-10,socy-10,socx+10,socy+10,fill="#ff7a45",outline="")
        else:
            self.canvas.coords(self.items["orange"],socx-10,socy-10,socx+10,socy+10)
        # forward direction lines
        if "fwd" in state["blue"]:
            fx,fy,_=state["blue"]["fwd"]
            endx,endy=self._map(bcx+fx*120,bcy+fy*120)
            if self.items["blue_dir"] is None:
                self.items["blue_dir"]=self.canvas.create_line(sbcx,sbcy,endx,endy,fill="#87d4ff",width=2)
            else:
                self.canvas.coords(self.items["blue_dir"],sbcx,sbcy,endx,endy)
        if "fwd" in state["orange"]:
            fx,fy,_=state["orange"]["fwd"]
            endx,endy=self._map(ocx+fx*120,ocy+fy*120)
            if self.items["orange_dir"] is None:
                self.items["orange_dir"]=self.canvas.create_line(socx,socy,endx,endy,fill="#ffc7a6",width=2)
            else:
                self.canvas.coords(self.items["orange_dir"],socx,socy,endx,endy)
        # boost pads (create once, update color only)
        pads=state.get("boost_pads",[])
        if not self.items["pads"]:
            for p in pads:
                px,py,_=p["pos"]
                spx,spy=self._map(px,py)
                r=6 if p["is_big"] else 4
                col="#2bd06a" if p["is_active"] else "#1b4d2f"
                pid=self.canvas.create_oval(spx-r,spy-r,spx+r,spy+r,fill=col,outline="")
                self.items["pads"].append((pid,p))
        else:
            for (pid,p) in self.items["pads"]:
                col="#2bd06a" if p.get("is_active",False) else "#1b4d2f"
                self.canvas.itemconfig(pid,fill=col)
        text=f"step {metrics.get('step',0)}  fps {metrics.get('fps',0)}  rew {metrics.get('recent_reward',0):.3f}  iters {metrics.get('iterations',0)}"
        self.info.config(text=text)
        # no sparkline to avoid flicker in lower area; consider a separate overlay if needed
    def _blend(self,a,b,alpha):
        return [a[i]*(1-alpha)+b[i]*alpha for i in range(3)]
    def loop(self):
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue
            try:
                m={} ; s=None
                if Path(self.metrics_path).exists():
                    mt=os.path.getmtime(self.metrics_path)
                    if mt!=self.last_metrics_mtime:
                        with open(self.metrics_path,"r") as f:
                            m=json.load(f)
                        self.last_metrics_mtime=mt
                if Path(self.match_path).exists():
                    st=os.path.getmtime(self.match_path)
                    if st!=self.last_match_mtime:
                        with open(self.match_path,"r") as f:
                            s=json.load(f)
                        self.prev_state=self.curr_state
                        self.curr_state=s
                        self.last_match_mtime=st
                base={"blue":{"pos":[0,0,0]},"orange":{"pos":[0,0,0]},"ball":{"pos":[0,0,0]},"boost_pads":[]}
                if self.prev_state and self.curr_state:
                    alpha=min(1.0,max(0.0,0.3))
                    blended={
                        "blue":{"pos":self._blend(self.prev_state["blue"]["pos"],self.curr_state["blue"]["pos"],alpha),"fwd":self.curr_state["blue"].get("fwd",[0,0,0])},
                        "orange":{"pos":self._blend(self.prev_state["orange"]["pos"],self.curr_state["orange"]["pos"],alpha),"fwd":self.curr_state["orange"].get("fwd",[0,0,0])},
                        "ball":{"pos":self._blend(self.prev_state["ball"]["pos"],self.curr_state["ball"]["pos"],alpha)},
                        "boost_pads":self.curr_state.get("boost_pads",[])
                    }
                    self.draw(blended,m)
                else:
                    self.draw(self.curr_state or base,m)
            except Exception:
                pass
            time.sleep(0.016)

def main():
    updater.check_for_update(__file__)
    if not (hasattr(sys, "base_prefix") and sys.prefix!=sys.base_prefix) and not os.getenv("VIRTUAL_ENV"):
        raise RuntimeError("venv required. Activate a virtual environment before running.")
    # auto-discover latest session under outputs/sessions
    base="outputs"
    sessions_dir=os.path.join(base,"sessions")
    metrics=os.path.join(base,"live_metrics.json")
    match=os.path.join(base,"live_match_state.json")
    try:
        if os.path.isdir(sessions_dir):
            entries=[(d, os.path.getmtime(os.path.join(sessions_dir,d))) for d in os.listdir(sessions_dir) if os.path.isdir(os.path.join(sessions_dir,d))]
            if entries:
                latest=sorted(entries,key=lambda x:x[1])[-1][0]
                metrics=os.path.join(sessions_dir,latest,"live_metrics.json")
                match=os.path.join(sessions_dir,latest,"live_match_state.json")
    except Exception:
        pass
    viewer=LiveViewer(metrics,match)
    viewer.root.mainloop()

if __name__=="__main__":
    main()
