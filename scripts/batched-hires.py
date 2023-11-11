# stable-diffusion-webui-batched-hires
import traceback
import torch
import numpy as np
import gradio as gr
from modules import scripts,shared
class BatchedHiresExtension(scripts.Script):
    def __init__(self,*ar,**kw):
        super().__init__()
    def title(self):
        return 'batched-hires'
    def show(self,is_img2img):
        return scripts.AlwaysVisible
    def ui(self, is_img2img):
        if is_img2img:
            return []
        with gr.Row(elem_id='batched_hires_row'):
            with gr.Accordion('Batched Hires (reduce Batch size for Hires. fix)',open=False,elem_id='batched_hires_accordion'):
              gr_size = gr.Slider(minimum=0,maximum=8,step=1,label='Hires. fix Batch size (used if less than main Batch size; 0 to disable, 1 is recommended) ',value=0,elem_id='batched_hires_size')
        return [gr_size]
    def process(self,p,gr_size=None):
        if (not gr_size) or (gr_size<1):
            return
        sample_hr_pass_old = getattr(p,'sample_hr_pass')
        def sample_hr_pass_new(samples,decoded_samples,seeds,subseeds,subseed_strength,prompts,*ar,**kw):
            batch = samples.shape[0] if samples is not None else 0
            if batch<=gr_size:
                return sample_hr_pass_old(samples,decoded_samples,seeds,subseeds,subseed_strength,prompts,*ar,**kw)
            decoded = None
            u_decoded_samples = (decoded_samples is not None) and (decoded_samples.shape[0]>1)
            u_seeds = (seeds is not None) and (len(seeds)>1)
            u_subseeds = (subseeds is not None) and (len(subseeds)>1)
            u_prompts = (prompts is not None) and (len(prompts)>1)
            old_negative_prompts = p.negative_prompts if (p.negative_prompts is not None) and (len(p.negative_prompts)>1) else None
            old_hr_prompts = p.hr_prompts if (p.hr_prompts is not None) and (len(p.hr_prompts)>1) else None
            old_hr_negative_prompts = p.hr_negative_prompts if (p.hr_negative_prompts is not None) and (len(p.hr_negative_prompts)>1) else None
            old_size = p.batch_size
            old_c = (p.c.shape,p.c.batch) if (p.c is not None) and (len(p.c.batch)>1) else None
            old_uc = p.uc if (p.uc is not None) and (len(p.uc)>1) else None
            old_all_seeds = p.all_seeds if (p.all_seeds is not None) and (len(p.all_seeds)>1) else None
            old_all_subseeds = p.all_subseeds if (p.all_subseeds is not None) and (len(p.all_subseeds)>1) else None
            old_all_prompts = p.all_prompts if (p.all_prompts is not None) and (len(p.all_prompts)>1) else None
            old_all_negative_prompts = p.all_negative_prompts if (p.all_negative_prompts is not None) and (len(p.all_negative_prompts)>1) else None
            old_all_hr_prompts = p.all_hr_prompts if (p.all_hr_prompts is not None) and (len(p.all_hr_prompts)>1) else None
            old_all_hr_negative_prompts = p.all_hr_negative_prompts if (p.all_hr_negative_prompts is not None) and (len(p.all_hr_negative_prompts)>1) else None
            left = 0
            try:
                while (left<batch) and (not shared.state.interrupted):
                    right = min(left+gr_size,batch)
                    print('Batched Hires - take [{}:{}] of [{}]'.format(left,right,batch))
                    p.batch_size = right-left
                    if old_c:
                        p.c.batch = old_c[1][left:right]
                        p.c.shape = (p.batch_size,)+old_c[0][1:]
                    if old_uc:
                        p.uc = old_uc[left:right]
                    if u_seeds:
                        p.seeds = seeds[left:right]
                    if u_subseeds:
                        p.subseeds = subseeds[left:right]
                    if u_prompts:
                        p.prompts = prompts[left:right]
                    if old_negative_prompts:
                        p.negative_prompts = old_negative_prompts[left:right]
                    if old_hr_prompts:
                        p.hr_prompts = old_hr_prompts[left:right]
                    if old_hr_negative_prompts:
                        p.hr_negative_prompts = old_hr_negative_prompts[left:right]
                    if old_all_seeds:
                        p.all_seeds = old_all_seeds[left:right]
                    if old_all_subseeds:
                        p.all_subseeds = old_all_subseeds[left:right]
                    if old_all_prompts:
                        p.all_prompts = old_all_prompts[left:right]
                    if old_all_negative_prompts:
                        p.all_negative_prompts = old_all_negative_prompts[left:right]
                    if old_all_hr_prompts:
                        p.all_hr_prompts = old_all_hr_prompts[left:right]
                    if old_all_hr_negative_prompts:
                        p.all_hr_negative_prompts = old_all_hr_negative_prompts[left:right]
                    p.hr_c = None
                    p.hr_uc = None
                    res = sample_hr_pass_old(
                        samples[left:right],
                        decoded_samples[left:right] if u_decoded_samples else decoded_samples,
                        p.seeds,
                        p.subseeds,
                        subseed_strength,
                        p.prompts,
                    *ar,**kw)
                    if decoded is None:
                        decoded = res
                    else:
                        decoded.extend(res)
                    left = right
            except:
                traceback.print_exc()
            if old_c:
                p.c.shape = old_c[0]
                p.c.batch = old_c[1]
            if old_uc:
                p.uc = old_uc
            if u_seeds:
                p.seeds = seeds
            if u_subseeds:
                p.subseeds = subseeds
            if u_prompts:
                p.prompts = prompts
            if old_negative_prompts:
                p.negative_prompts = old_negative_prompts
            if old_hr_prompts:
                p.hr_prompts = old_hr_prompts
            if old_hr_negative_prompts:
                p.hr_negative_prompts = old_hr_negative_prompts
            if old_all_seeds:
                p.all_seeds = old_all_seeds
            if old_all_subseeds:
                p.all_subseeds = old_all_subseeds
            if old_all_prompts:
                p.all_prompts = old_all_prompts
            if old_all_negative_prompts:
                p.all_negative_prompts = old_all_negative_prompts
            if old_all_hr_prompts:
                p.all_hr_prompts = old_all_hr_prompts
            if old_all_hr_negative_prompts:
                p.all_hr_negative_prompts = old_all_hr_negative_prompts
            p.batch_size = old_size
            if decoded is None:
                return samples
            return decoded
        setattr(p,'sample_hr_pass',sample_hr_pass_new)
#EOF
