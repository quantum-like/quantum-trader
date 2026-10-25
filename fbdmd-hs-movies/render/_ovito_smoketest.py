import numpy as np, time, os
from ovito.vis import Viewport, TachyonRenderer
from ovito.pipeline import StaticSource, Pipeline
from ovito.data import DataCollection
rng=np.random.default_rng(0)
N=1000; L=11.44
pos=rng.uniform(0,L,(N,3))
data=DataCollection()
cell=data.create_cell([[L,0,0,0],[0,L,0,0],[0,0,L,0]], pbc=(True,True,True))
cell.vis.line_width=0.04; cell.vis.rendering_color=(0.35,0.35,0.4)
p=data.create_particles(count=N)
p.create_property('Position', data=pos)
types=p.create_property('Particle Type', data=np.r_[np.zeros(N-40,int)+1, np.ones(20,int)*2, np.ones(20,int)*3])
for tid,name,col in [(1,'HS',(0.78,0.78,0.80)),(2,'plus',(0.85,0.15,0.15)),(3,'minus',(0.15,0.35,0.9))]:
    t=types.add_type_id(tid,p,name=name); t.color=col; t.radius=0.5
pipe=Pipeline(source=StaticSource(data=data)); pipe.add_to_scene()
vp=Viewport(type=Viewport.Type.Perspective, camera_dir=(-1,-1.2,-0.8))
vp.zoom_all(size=(1280,720))
t0=time.time()
vp.render_image(size=(1280,720), filename='/tmp/ovtest/test.png', renderer=TachyonRenderer(ambient_occlusion=True, shadows=True, antialiasing=True), background=(0.97,0.97,0.98))
print('render seconds', time.time()-t0)
