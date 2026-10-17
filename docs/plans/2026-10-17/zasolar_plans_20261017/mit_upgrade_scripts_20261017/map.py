import torch,re,json,collections
from transformers import SegformerConfig,SegformerForSemanticSegmentation,PvtV2Config,PvtV2Model
sd=torch.load('/home/gao/capetown_eval/ds3m_ft/run_v2d1/best.pt',map_location='cpu')
if 'model' in sd: sd=sd['model']
print('n_tensors',len(sd),'params',sum(v.numel() for v in sd.values()))
print('prefixes',collections.Counter(k.split('.')[0]+'.'+k.split('.')[1] for k in sd))
print('classifier',sd['decode_head.classifier.weight'].shape)
def cmp(src,dst):
    m=mis=sh=0
    for k,v in dst.items():
        if k not in src: mis+=1
        elif src[k].shape!=v.shape: sh+=1
        else: m+=1
    unexp=len([k for k in src if k not in dst]); return dict(matched=m,missing=mis,shape_mismatch=sh,unexpected=unexp,dst_total=len(dst),
      matched_params=sum(v.numel() for k,v in dst.items() if k in src and src[k].shape==v.shape),dst_params=sum(v.numel() for v in dst.values()))
R={}
for name,d in [('b3',[3,4,18,3]),('b4',[3,8,27,3])]:
    t=SegformerForSemanticSegmentation(SegformerConfig(num_labels=3,depths=d,hidden_sizes=[64,128,320,512],decoder_hidden_size=768)).state_dict()
    R[name]=cmp(sd,t)
    # depth expansion: map new block j -> b2 block j % depth
    src2=dict(sd); b2d=[3,4,6,3]
    for k in t:
        mm=re.match(r'(segformer\.encoder\.block\.(\d)\.)(\d+)(\..*)',k)
        if mm and k not in sd:
            s,j=int(mm.group(2)),int(mm.group(3)); kk=f'{mm.group(1)}{j%b2d[s]}{mm.group(4)}'
            if kk in sd: src2[k]=sd[kk]
    R[name+'_dupblocks']=cmp(src2,t)
pv=PvtV2Model(PvtV2Config(depths=[3,4,6,3],hidden_sizes=[64,128,320,512],num_attention_heads=[1,2,5,8],sr_ratios=[8,4,2,1],mlp_ratios=[8,8,4,4])).state_dict()
enc={k[len('segformer.'):]:v for k,v in sd.items() if k.startswith('segformer.')}
R['pvt_raw']=cmp(enc,pv)
def rn(k):
    k=k.replace('encoder.patch_embeddings.','encoder.layers.X.')
    return k
# build mapping by structural rename
S={}
for k,v in enc.items():
    m=re.match(r'encoder\.patch_embeddings\.(\d)\.(proj|layer_norm)\.(.*)',k)
    if m: S[f'encoder.layers.{m[1]}.patch_embedding.{m[2]}.{m[3]}']=v; continue
    m=re.match(r'encoder\.block\.(\d)\.(\d+)\.(.*)',k)
    if m:
        r=m[3].replace('attention.self.','attention.').replace('attention.output.dense','attention.proj').replace('mlp.dense1','mlp.dense1').replace('mlp.dwconv.dwconv','mlp.dwconv.dwconv')
        S[f'encoder.layers.{m[1]}.blocks.{m[2]}.{r}']=v; continue
    m=re.match(r'encoder\.layer_norm\.(\d)\.(.*)',k)
    if m: S[f'encoder.layers.{m[1]}.layer_norm.{m[2]}']=v; continue
    S[k]=v
R['pvt_remap']=cmp(S,pv)
R['pvt_unmatched_dst_examples']=[k for k in pv if k not in S or S[k].shape!=pv[k].shape][:15]
R['pvt_dst_examples']=list(pv)[:12]
print(json.dumps(R,indent=1))
