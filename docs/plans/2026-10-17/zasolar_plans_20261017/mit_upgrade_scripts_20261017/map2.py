import torch,re,json
from transformers import SegformerConfig,SegformerForSemanticSegmentation,PvtV2Config,PvtV2Model
sd=torch.load('/home/gao/capetown_eval/ds3m_ft/run_v2d1/best.pt',map_location='cpu')
def cmp(src,dst):
    ok=[k for k in dst if k in src and src[k].shape==dst[k].shape]
    return dict(matched=len(ok),missing=sum(k not in src for k in dst),shape_mismatch=sum(k in src and src[k].shape!=dst[k].shape for k in dst),
      unexpected=sum(k not in dst for k in src),dst_total=len(dst),matched_params=sum(dst[k].numel() for k in ok),dst_params=sum(v.numel() for v in dst.values()))
R={}
for name,d in [('b3',[3,4,18,3]),('b4',[3,8,27,3])]:
    t=SegformerForSemanticSegmentation(SegformerConfig(num_labels=3,depths=d,hidden_sizes=[64,128,320,512],decoder_hidden_size=768)).state_dict()
    s2=dict(sd); b2=[3,4,6,3]
    for k in t:
        m=re.match(r'(segformer\.stages\.(\d)\.blocks\.)(\d+)(\..*)',k)
        if m and k not in sd: s2[k]=sd[f'{m[1]}{int(m[3])%b2[int(m[2])]}{m[4]}']
    R[name+'_dupblocks']=cmp(s2,t)
pvm=PvtV2Model(PvtV2Config(depths=[3,4,6,3],hidden_sizes=[64,128,320,512],num_attention_heads=[1,2,5,8],sr_ratios=[8,4,2,1],mlp_ratios=[8,8,4,4]))
pv=pvm.state_dict(); print('pvt keys sample',[k for k in pv if 'layers.0.blocks.0' in k])
S={}
rep=[('patch_embeddings.','patch_embedding.'),('layernorm_before','layer_norm_1'),('layernorm_after','layer_norm_2'),('q_proj','query'),('k_proj','key'),('v_proj','value'),('o_proj','proj'),
     ('sequence_reduction.sequence_reduction','spatial_reduction'),('sequence_reduction.layer_norm','layer_norm'),('mlp.fc1','mlp.dense1'),('mlp.fc2','mlp.dense2')]
for k,v in sd.items():
    if not k.startswith('segformer.stages'): continue
    kk=k.replace('segformer.stages.','encoder.layers.')
    for a,b in rep: kk=kk.replace(a,b)
    S[kk]=v
R['pvt_remap']=cmp(S,pv); R['pvt_unmatched']=[(k,tuple(pv[k].shape),tuple(S[k].shape) if k in S else None) for k in pv if k not in S or S[k].shape!=pv[k].shape][:20]
R['pvt_leftover_src']=[k for k in S if k not in pv][:10]
print(json.dumps(R,indent=1,default=str))
