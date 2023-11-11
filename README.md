# batched-hires

This is Extension for [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) to set the batch size for second hires.fix pass separately from the first pass.

## Installation:
Copy the link to this repository into `Extension index URL` in WebUI Extensions tab:
```
https://github.com/klimaleksus/stable-diffusion-webui-batched-hires
```
Also you may clone/download this repository and put it to `stable-diffusion-webui/extensions` directory.

## Usage:
You will see a section titled `Batched Hires` on txt2img tab. It has only one slider that sets your desired batch size for high-resolution step of hires.fix.

This extension will not do anything if:
- You are not in `txt2img`; or
- You are not using `Hires. fix`; or
- Your main batch size is less or equal to `Batched Hires` value of this extension; or
- `Batched Hires` value is 0 (default).

Otherwise, this extension hooks your high-resolution pass to split the current batch into smaller chunks of your chosen value (the last one may be even smaller if numbers are not divisible).

I recommend to set Batched Hires slider to 1 and keep it there if you have low VRAM or use SDXL. To save this value for all future runs, you can reload your browser tab with WebUI, then change Batched Hires, and then go to `Settings` → `Default`, press `View changes` and `Apply`

## When you might need this:
Ever notice that you can cook large batches with pure txt2img in low resolution, but you have to use batch size of just 1 or 2 when upscaling in img2img?

This is because big images cannot fit into your GPU (especially when using SDXL) if you are batching them together.
It is not a problem if you want to generate a lot of drafts in low resolution with txt2img, and then grab the best one to upscale further.

But for Hires.fix it is a problem! Because with hires-fix enabled you cannot use large batches anymore, since the second high-resolution step will inherit the same batch size, most likely resulting in CUDA out of memory.  
(Buy the way, [Tiled VAE](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111) can help you to deal with large resolutions if you see error at 100% of your generation progress; be sure to enable `Tiled VAE`, not `Tiled Diffusion` there!)

With this extension you can perform Hires step will small batches while still using larger batch sizes in low resolution. Just set `Batched Hires` to 1 (or how large your batches usually are when you upscaling in img2img), it does not matter if the extension section opened or not.

## How it works:
The extension hooks `p.sample_hr_pass` function at `scripts.process` callback at the beginning of generation if `Batched Hires` is not zero. Hooking is done at local instance level.

Then, when the hooked version gets control (presumingly after the low resolution pass is already done, and a high-resolution pass of Hires.fix is upcoming), it checks if the main batch size is indeed larger than `Batched Hires`; if not, it calls the original version and exists.

Before looping for split batches, the extension saves these variables:
```
samples
decoded_samples
seeds
subseeds
prompts
p.negative_prompts
p.hr_prompts
p.hr_negative_prompts
p.batch_size
p.c
p.uc
p.all_seeds
p.all_subseeds
p.all_prompts
p.all_negative_prompts
p.all_hr_prompts
p.all_hr_negative_prompts
```

Whatever of that is detected as being batched – will be sliced before calling the original sample_hr_pass() function several times. Before each call, the extension rewrites `p.batch_size`, `p.prompts`, and deletes `p.hr_c` with `p.hr_uc` for them to be recalculated by WebUI later.

If an exception occurs, or if the user aborts the generation, the loop breaks with whatever images were done (eating all low resolution copies, to not return images with different resolutions together).
Before returning the control down the line, this extension restores all variables that were changed.

## Compatibility:
This extension may cause troubles for any other extension that:
- don't expect `scripts.before_hr` to be called several times;
- hooks samplers, individual steps or inner loop assuming something for hires-pass before it's actually done;
- caches copies of `p.*` variables that this extension rewrites, instead of using their actual values.

If you have problems with other extensions that you have to use – welp, you will need to lower your main batch size to match `Batched Hires` (effectively disabling this extension).  
Also you may create and Issue in this repo, just be sure to provide an example how to see the error, and which extension or setting is causing it. Maybe that other extension could be tweaked to support batched hires too!

Your total generation progress (shown in browser) will lie if this extension did the splitting. Because the progress bar will think that high resolution pass is fully done at the very first sub-batch, and then it will show around 99% till the end.  
I'll consider hooking it later.

Seeds should not break, no matter which batched-hires value you will use.  
But in practice, small differences may appear. Those differences sometimes appear even with this extension not installed, so I cannot tell for sure what's causing them, probably just ordinary floating point rounding errors as always.

This extension does not add anything to generation info.  
It prints `Batched Hires - take [X:Y] of [Z] ` to console before invoking hires when in effect; where Z is your lowres batch size, and X:Y is the slice of it that will be used in this iteration (with the new batch size being Y-X).
### EOF
