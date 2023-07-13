# X Server

## 1. Function

### 1.1 Timestamps Alignment

Merge data from real-time data stream (gateway and optitrack).

### 1.2 Deploy localization Algorithm

Fusion MUSIC, Tagoram....



## Data format

### Real-time data stream

#### From gateway

```python
#from gateway
{
  time_today:{
    'version':v1,                   # The hardware version
    'tagid':tagid,           
  	'phase': phase,                 # phase
    "rss":[r1,r2,..., r16],					# Receive signal strength
    "createdTime": us               # The time stamp received by the gateway
    "sourcefile": filename          # original filename
  }
}
```

#### From Optitrack (if have)

```python
#from optitrack
{
  time_today:{
    'gateway1':[[x1,x2,x3,x4],[y1,y2,y3,y4],[z1,z2,z3,z4]],
    'gateway2':[[x1,x2,x3,x4],[y1,y2,y3,y4],[z1,z2,z3,z4]],
    'gateway3':[[x1,x2,x3,x4],[y1,y2,y3,y4],[z1,z2,z3,z4]],
    'target':[x,y,z]
  }
}
```

#### From Optitrack (if not have)

```python
#from optitrack
{
  time_today:{
    'gateway1':[[0,0,0,0],[0,0,0,0],[0,0,0,0]],
    'gateway2':[[0,0,0,0],[0,0,0,0],[0,0,0,0]],
    'gateway3':[[0,0,0,0],[0,0,0,0],[0,0,0,0]],
    'target':[0,0,0]
  }
}
```



#### To main server

```python
# To server main
{
  "logTime":us,        # the logical time after the queque alignment
  "phyTime":us,        # the real timestamp created by the xserver 
  'gateway1':{
    'version':v1,                   # The hardware version
    'tagid':tagid,           
  	'phase': phase,                 # phase
    "rss":[r1,r2,..., r16],					# Receive signal strength
    "createdTime": us               # The time stamp received by the gateway
    "sourcefile": filename          # original filename
  },
  'gateway2':{
    'version':v1,                   # The hardware version
    'tagid':tagid,           
  	'phase': phase,                 # phase
    "rss":[r1,r2,..., r16],					# Receive signal strength
    "createdTime": us               # The time stamp received by the gateway
    "sourcefile": filename          # original filename
  },
  'gateway3':{
    'version':v1,                   # The hardware version
    'tagid':tagid,           
  	'phase': phase,                 # phase
    "rss":[r1,r2,..., r16],					# Receive signal strength
    "createdTime": us               # The time stamp received by the gateway
    "sourcefile": filename          # original filename
  },
  'optitrack':{
    'gateway1':[[x1,x2,x3,x4],[y1,y2,y3,y4],[z1,z2,z3,z4]],
    'gateway2':[[x1,x2,x3,x4],[y1,y2,y3,y4],[z1,z2,z3,z4]],
    'gateway3':[[x1,x2,x3,x4],[y1,y2,y3,y4],[z1,z2,z3,z4]],
    'target':[x,y,z]    
  }
}
```



### Display data stream

```python
# To GUI
{
    "tagId":"xxxx",
    "logTime":us,        # the logical time after the queque alignment
    "phyTime":us,        # the real timestamp created by the xserver 
    "position": [x,y,z], # the computed position by xserver.
    "truth":[x,y,z],     # computed by optiTrack
    "xServer":{
        "frequency":hz,  # the sniffer frequency
        "gateways":{
            "gateway1":{ # gateway's id
                "version":v1,                   # The hardware version
                "position":[p1,p2,p3,p4],       # The position of the gateway.
                "phase":[p1,p1,....,p16],       # The calibrated phase 
                "phaseOffset":[o1,o2,....,o16], # The offset for the calibration
                "rss":[r1,r2,..., r16],					# Receive signal strength
                "aoa":{                         # for the radar map
                    "azimuth":angle,
                    "elevation":angle,
                }，
                "createdTime": us               # The time stamp received by the gateway
                "sourcefile": filename          # original filename
            },
            "gateway2":{
               "version":v1,
                "position":[p1,p2,p3,p4],
                "phase":[p1,p1,....,p16],
                "phaseOffset":[o1,o2,....,o16],      
                "rss":[r1,r2,..., r16],
                "aoa":{                         # for the radar map
                    "azimuth":angle,
                    "elevation":angle,
                }，
                "createdTime": us
                "sourcefile": filename          # original filename
            }
        }
    },
    "spectrum":{
        "algorithm": music,       # which algorithm is used.
        "confidence": x,          # how many gateways contribute to this spectrum
        "xRange":[x1,x2],         # searched range
        "yRange":[y1,y2],         # searched range
        "zRange":[z1,z2],         # searched range. If z1=z2, the searching is taken on a single plane.
        "data":[b1,b2,b3.....b1024],  # spectrum image
        "createdTime"：us             # the timestap created.
    }
}

```





