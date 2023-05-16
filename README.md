# llm

Script Options

You can customize this script with several command-line arguments to tailor the results to what you want. Let's take a look some that might come in handy:

    --prompt followed by a sentence in quotation marks will specify the prompt to generate the image for. The default is "a painting of a virus monster playing guitar".
    --from-file specifies a filepath for a file of prompts to use to generate images for.
    --ckpt followed by a path specifies which checkpoint of model to use. The default is models/ldm/stable-diffusion-v1/model.ckpt.
    --outdir followed by a path will specify the output directory to save the generate image to. The default is outputs/txt2img-samples.
    --skip_grid will skip creating an image that combines.
    --ddim_steps followed by an integer specifies the number of sampling steps in the Diffusion process. Increasing this number will increase computation time but may improve results. The default value is 50.
    --n_samples followed by an integer specifies how many samples to produce for each given prompt (the batch size). The default value is 3.
    --n_iter followed by an integer specifies how many times to run the sampling loop. Effectively the same as --n_samples, but use this instead if running into OOM error. See the source code for clarification. The default value is 2.
    --H followed by an integer specifies the height of the generated images (in pixels). The default value is 512.
    --W followed by an integer specifies the width of generated images (in pixels). The default value is 512.
    --scale followed by a float specifies the guidance scale to use. The default value is 7.5
    --seed followed by an integer allows for setting the random seed (for reproducible results). The default value is 42.

You can see a full list of possible arguments with default values in the txt2img.py file. Let's see a more complicated generation prompt using these optional arguments now.

In the stable-diffusion directory, create a file called prompts.txt. Create several prompts, one on each line of the file. For example:

A dolphin flying a plane

python scripts/txt2img.py \
--from-file prompts.txt \
--ckpt sd-v1-4.ckpt \
--outdir generated-images \
--skip_grid \
--ddim_steps 100 \
--n_iter 3 \
--H 256 \
--W 512 \
--n_samples 3 \
--scale 8.0 \
--seed 119

Errors:ImportError: cannot import name 'SAFE_WEIGHTS_NAME' from 'transformers.utils'.

Solution: Edit environments.yaml, "diffusers" --> "diffusers==0.12.1", then conda env update -f environment.yaml

Error:SSLError: HTTPSConnectionPool(host='huggingface.co', port=443) Solution: set os.environ['CURL_CA_BUNDLE'] = ''
