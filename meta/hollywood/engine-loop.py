import numpy as np, soundfile as sf, os
T=os.environ["TEMP"]
x,sr=sf.read(os.path.join(T,"eng2202.wav")); m=x[sr:-sr].mean(1)
# the take's average spectrum (Welch, 16k windows): its engine harmonics and rumble, with every tick averaged away
W=16384; win=np.hanning(W); segs=[m[i:i+W]*win for i in range(0,len(m)-W,W//4)]
mag=np.sqrt(np.mean([np.abs(np.fft.rfft(s))**2 for s in segs],0))
N=8*sr  # one cycle; a single inverse FFT is periodic over N, so the loop has no seam to hear
f=np.fft.rfftfreq(N,1/sr); spec=np.interp(f,np.fft.rfftfreq(W,1/sr),mag)
spec*=1/(1+(f/4000)**6); spec[f<25]=0
rng=np.random.default_rng(7)
y=np.fft.irfft(spec*np.exp(2j*np.pi*rng.random(len(f))),N)
# the cylinders' pulse: a light tremble at the firing rate, a whole number of cycles per loop to stay seamless
k=round(28*N/sr); y*=1+0.12*np.sin(2*np.pi*k*np.arange(N)/N)
y*=0.7/np.abs(y).max()
c=np.stack([y,y],1); P=int(0.2*sr)
sf.write(os.path.join(T,"loop.wav"),np.concatenate([c[-P:],c,c[:P]]),sr); print(N)
