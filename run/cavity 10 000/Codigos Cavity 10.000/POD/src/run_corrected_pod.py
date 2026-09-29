"""POD corrigida para cavity: Ux, Uy, p, T e k, sem zeros silenciosos.

Usa matriz float32 em memmap e SVD aleatorizada truncada para evitar a SVD
quadrada de 1,9 GB produzida pelo notebook legado. Salva somente os objetos
necessários para reconstrução e POD--SINDy.
"""
from __future__ import annotations
import json, shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
from fluidfoam import readfield, readmesh
from sklearn.utils.extmath import randomized_svd

CASE=Path(r'/home/mlep-pc-ubuntu/OpenFOAM/mlep-pc-ubuntu-13/run/cavity 10 000')
ROOT=CASE/'Codigos Cavity 10.000/POD'
OUT=ROOT/'resultados_corrigidos'; OUT.mkdir(parents=True,exist_ok=True)
RANK=50; SEED=7
CHANNELS=['Ux','Uy','p','T','k']

def times():
 out=[]; rejected=[]
 required=('U','p','T','k')
 for path in CASE.iterdir():
  if not path.is_dir(): continue
  try: float(path.name)
  except ValueError: continue
  missing=[name for name in required if not (path/name).exists()]
  if missing: rejected.append({'time':path.name,'missing':missing})
  else: out.append(path.name)
 return sorted(out,key=float), sorted(rejected,key=lambda row:float(row['time']))

def uv(t,n):
 a=np.asarray(readfield(str(CASE),t,'U'),dtype=np.float32)
 if a.shape==(3,n): return a[0],a[1]
 if a.shape==(n,3): return a[:,0],a[:,1]
 if a.size==3*n:
  a=a.reshape(n,3); return a[:,0],a[:,1]
 raise ValueError(f'U inválido em {t}: {a.shape}')

def scalar(t,name,n):
 try: a=np.asarray(readfield(str(CASE),t,name),dtype=np.float32).squeeze()
 except Exception as e: raise FileNotFoundError(f'Campo obrigatório {name} ausente em {t}') from e
 if a.ndim==0:return np.full(n,float(a),np.float32)
 if a.size==n:return a.reshape(-1)
 raise ValueError(f'{name} inválido em {t}: {a.shape}')

def main():
 ts,rejected=times(); tv=np.asarray(list(map(float,ts)))
 print(f'snapshots válidos={len(ts)} rejeitados={len(rejected)}: {rejected}',flush=True)
 x,y,_=readmesh(str(CASE)); x=np.asarray(x).ravel(); y=np.asarray(y).ravel(); n=x.size
 shape=(len(CHANNELS)*n,len(ts)); mmap_path=OUT/'snapshots_centered.float32'
 X=np.memmap(mmap_path,dtype='float32',mode='w+',shape=shape)
 for j,t in enumerate(ts):
  ux,uy=uv(t,n); vals=[ux,uy,scalar(t,'p',n),scalar(t,'T',n),scalar(t,'k',n)]
  for i,v in enumerate(vals):X[i*n:(i+1)*n,j]=v
  if j%250==0: print(f'leitura {j}/{len(ts)} t={t}',flush=True)
 X.flush(); mean=np.asarray(np.mean(X,axis=1,dtype=np.float64),dtype=np.float32)
 for j in range(shape[1]): X[:,j]-=mean
 X.flush(); total=float(np.sum(np.asarray(X,dtype=np.float64)**2)); print('SVD...',flush=True)
 U,s,Vt=randomized_svd(X,n_components=RANK,n_iter=7,random_state=SEED)
 A=s[:,None]*Vt
 energy=np.cumsum(s.astype(np.float64)**2)/total
 errors=np.sqrt(np.maximum(0,1-energy))
 np.savez_compressed(OUT/'pod_results.npz',times=np.asarray(ts),x=x,y=y,U=U,s=s,Vt=Vt,A=A,
  Xmean=mean,centered=np.asarray([True]),channel_names=np.asarray(CHANNELS,dtype=object),
  field_names=np.asarray(['U','p','T','k'],dtype=object),nCells=np.asarray([n]),nChannels=np.asarray([5]),
  total_energy=np.asarray([total]),energy_cum=energy,err_r=errors,r_final=np.asarray([RANK]),
  normalization=np.asarray(['none'],dtype=object),source_case=np.asarray([str(CASE)],dtype=object))
 plt.figure();plt.plot(range(1,RANK+1),energy,'o-');plt.xlabel('modos');plt.ylabel('energia acumulada');plt.grid(alpha=.3);plt.tight_layout();plt.savefig(OUT/'energia_corrigida.png',dpi=180);plt.close()
 plt.figure();plt.semilogy(range(1,RANK+1),errors,'o-');plt.xlabel('modos');plt.ylabel('erro global estimado');plt.grid(alpha=.3);plt.tight_layout();plt.savefig(OUT/'erro_corrigido.png',dpi=180);plt.close()
 # modos 1 e 2 na malha estruturada 40x40
 for mode in range(2):
  fig,ax=plt.subplots(1,5,figsize=(17,3.2))
  for i,name in enumerate(CHANNELS):
   im=ax[i].imshow(U[i*n:(i+1)*n,mode].reshape(40,40),origin='lower',cmap='coolwarm');ax[i].set_title(f'modo {mode+1}: {name}');ax[i].axis('off');fig.colorbar(im,ax=ax[i],fraction=.046,pad=.04)
  fig.tight_layout();fig.savefig(OUT/f'modo_{mode+1:02d}.png',dpi=180);plt.close(fig)
 meta={'n_snapshots':len(ts),'n_cells':n,'channels':CHANNELS,'time_min':float(tv.min()),'time_max':float(tv.max()),
  'rank':RANK,'energy_at_8':float(energy[7]),'energy_at_16':float(energy[15]),'energy_at_50':float(energy[-1]),
  'error_at_8':float(errors[7]),'error_at_16':float(errors[15]),'error_at_50':float(errors[-1]),
  'normalization':'none','centered':True,'algorithm':'randomized_svd','n_iter':7,'seed':SEED,
  'rejected_snapshots':rejected}
 (OUT/'summary.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 del X; mmap_path.unlink(missing_ok=True); print(json.dumps(meta,indent=2),flush=True)
if __name__=='__main__':main()
