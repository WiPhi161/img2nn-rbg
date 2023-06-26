# img2nn-rbg
A neural network that interpolates two colour images using Tsoding's [nn.h](https://github.com/tsoding/nn.h) framework. A continuation of the grayscale image model develop in [Tsoding's neutral network series](https://www.twitch.tv/videos/1824829365)

![Project Screenshot](https://github.com/WiPhi161/img2nn-rbg/assets/100654058/33324452-690a-4e73-929c-7de08a227fb4)

## Quickstart
```console
$ ./build.sh
$ ./img_rgb img1.png img2.png
````
## Usage
Press SPACE to start/stop training the model
Press R to randomise the model's weights and biases
Press S to take an unscaled screenshot of the preview model output
## Results

https://github.com/WiPhi161/img2nn-rbg/assets/100654058/9d646e80-ac53-442a-bf3f-6769f6a2d72c

This is what the training process looks like.

After about 7000 epochs here are the final upscaled images of input image 1, input image 2, and the interpolated "in-between" image.
![upscaled](https://github.com/WiPhi161/img2nn-rbg/assets/100654058/d172d210-42a7-4279-9b51-532572a5ec06)
![upscaled_3](https://github.com/WiPhi161/img2nn-rbg/assets/100654058/2c10d42c-649d-4339-b10d-a6ec8c77d56b)
![upscaled2](https://github.com/WiPhi161/img2nn-rbg/assets/100654058/62e99c75-b596-4f9e-93c0-1e624b1b6d76)

## Credits
Tsoding and his awesome nn library: (https://github.com/tsoding/nn.h)

Example Images Source: [GrafxKid](https://opengameart.org/content/classic-hero-and-baddies-pack)





