#include <assert.h>
#include <stdio.h>
#include <float.h>
#include <errno.h>
#include <string.h>

#include <raylib.h>
#include <raymath.h>

#include <sys/types.h>
// #include <sys/wait.h>
#include <unistd.h>

#include "stb_image.h"
#include "stb_image_write.h"

#define NN_IMPLEMENTATION
#define NN_ENABLE_GYM
#include "nn.h"

size_t arch[] = {3, 10,10, 3}; //last layer three because RBG
size_t max_epoch = 100*1000;
size_t batches_per_frame = 200;
size_t batch_size = 28;
float rate = 1.0f;
float scroll = 0.f;
bool paused = true;
#define NUM_COMPONENETS 4

#define out_width 512*2
#define out_height 512*2
uint32_t out_pixels[out_width*out_height];


char *args_shift(int *argc, char ***argv)
{
    assert(*argc > 0);
    char *result = **argv;
    (*argc) -= 1;
    (*argv) += 1;
    return result;
}

typedef enum {
    GHA_LEFT,
    GHA_RIGHT,
    GHA_CENTER,
} Gym_Horz_Align;

typedef enum {
    GVA_TOP,
    GVA_BOTTOM,
    GVA_CENTER,
} Gym_Vert_Align;

Vector2 render_texture_in_slot(Texture2D texture, Gym_Horz_Align ha, Gym_Vert_Align va, Gym_Rect r)
{
    Vector2 position = {r.x, r.y};
    float scale = 0;
    if (r.w > r.h) {
        scale = r.h/texture.height;
        switch (ha) {
        case GHA_LEFT: break;
        case GHA_RIGHT:
            position.x += r.w;
            position.x -= texture.width*scale;
            break;
        case GHA_CENTER:
            position.x += r.w/2;
            position.x -= texture.width*scale/2;
            break;
        }        
        DrawTextureEx(texture, position, 0, scale, WHITE);
    } else {
        scale = r.w/texture.width;
        switch (va) {
        case GVA_TOP: break;
        case GVA_BOTTOM:
            position.y += r.h;
            position.y -= texture.height*scale;
            break;
        case GVA_CENTER:
            position.y += r.h/2;
            position.y -= texture.height*scale/2;
            break;
        }
        DrawTextureEx(texture, position, 0, scale, WHITE);
    }

    Vector2 mouse_position = GetMousePosition();
    Rectangle hitbox = {
        position.x,
        position.y,
        texture.width*scale,
        texture.height*scale,
    };
    return CLITERAL(Vector2) {
        (mouse_position.x - position.x)/hitbox.width,
        (mouse_position.y - position.y)/hitbox.height
    };
}

void gym_nn_image_rgb(NN nn, void *pixels, size_t width, size_t height, size_t stride, float low, float high)
{
    NN_ASSERT(NN_INPUT(nn).cols >= 2);
    NN_ASSERT(NN_OUTPUT(nn).cols >= 1);
    uint32_t *pixels_u32 = pixels;
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(float)(width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(float)(height - 1);
            nn_forward(nn);

            float r = MAT_AT(NN_OUTPUT(nn), 0, 0);
            if (r < low) r = low;
            if (r > high) r = high;
            float g = MAT_AT(NN_OUTPUT(nn), 0, 1);
            if (g < low) g = low;
            if (g > high) g = high;
            float b = MAT_AT(NN_OUTPUT(nn), 0, 2);
            if (b < low) b = low;
            if (b > high) b = high;

            uint32_t rr = (r + low)/(high - low)*255.f;
            uint32_t gg = (g + low)/(high - low)*255.f;
            uint32_t bb = (b + low)/(high - low)*255.f;
            pixels_u32[y*stride + x] = (0xFF<<(8*3))|(bb<<(8*2))|(gg<<(8*1))|(rr<<(8*0));
        }
    }
}
void render_single_out_image(NN nn, float a)
{
    for (size_t i = 0; i < out_width*out_height; ++i) {
        out_pixels[i] = 0xFF000000;
    }

    size_t px, py, size;
    if (out_width > out_height) {
        size = out_height;
        px = out_width/2 - size/2;
        py = 0;
    } else {
        size = out_width;
        px = 0;
        py = out_height/2 - size/2;
    }

    MAT_AT(NN_INPUT(nn), 0, 2) = a;
    gym_nn_image_rgb(nn, &out_pixels[py*out_width + px], size, size, out_width, 0, 1);
}

int render_upscaled_screenshot(NN nn, const char *out_file_path)
{
    render_single_out_image(nn, scroll);

    if (!stbi_write_png(out_file_path, out_width, out_height, 4, out_pixels, out_width*sizeof(*out_pixels))) {
        fprintf(stderr, "ERROR: could not save image %s\n", out_file_path);
        return 1;
    }

    printf("Generated %s\n", out_file_path);
    return 0;
}
int main(int argc, char **argv){
	const char *program = args_shift(&argc, &argv);

	if (argc <= 0) {
	    fprintf(stderr, "Usage: %s <image1> <image2>\n", program);
	    fprintf(stderr, "ERROR: no image1 is provided\n");
	    return 1;
	}
	const char *img1_file_path = args_shift(&argc, &argv);

	if (argc <= 0) {
	    fprintf(stderr, "Usage: %s <image1> <image2>\n", program);
	    fprintf(stderr, "ERROR: no image2 is provided\n");
	    return 1;
	}
	const char *img2_file_path = args_shift(&argc, &argv);

	int img1_width, img1_height, img1_comp;
	uint32_t *img1_pixels = (uint32_t *)stbi_load(img1_file_path, &img1_width, &img1_height, &img1_comp, 0);
	if (img1_pixels == NULL) {
	    fprintf(stderr, "ERROR: could not read image %s\n", img1_file_path);
	    return 1;
	}
	if (img1_comp != NUM_COMPONENETS) {
	    fprintf(stderr, "ERROR: %s is %d bits image. Only 8 bit grayscale images are supported\n", img1_file_path, img1_comp*8);
	    return 1;
	}

	int img2_width, img2_height, img2_comp;

	uint32_t *img2_pixels = (uint32_t *)stbi_load(img2_file_path, &img2_width, &img2_height, &img2_comp, 0);
	if (img2_pixels == NULL) {
	    fprintf(stderr, "ERROR: could not read image %s\n", img2_file_path);
	    return 1;
	}
	if (img2_comp != NUM_COMPONENETS) {
	    fprintf(stderr, "ERROR: %s is %d bits image. Only 8 bit grayscale images are supported\n", img2_file_path, img2_comp*8);
	    return 1;
	}

	printf("%s size %dx%d %d bits\n", img1_file_path, img1_width, img1_height, img1_comp*8);
	printf("%s size %dx%d %d bits\n", img2_file_path, img2_width, img2_height, img2_comp*8);

    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));

    Mat t = mat_alloc(img1_width*img1_height + img2_width*img2_height, NN_INPUT(nn).cols + NN_OUTPUT(nn).cols);
    for (int y = 0; y < img1_height; ++y) {
        for (int x = 0; x < img1_width; ++x) {
            size_t i = y*img1_width + x;
            MAT_AT(t, i, 0) = (float)x/(img1_width - 1);
            MAT_AT(t, i, 1) = (float)y/(img1_height - 1);
            MAT_AT(t, i, 2) = 0.f;
            MAT_AT(t, i, 3) = (img1_pixels[i]&0x000000ff)/255.f;
            MAT_AT(t, i, 4) = ((img1_pixels[i]&0x0000ff00)>>8)/255.f;
            MAT_AT(t, i, 5) = ((img1_pixels[i]&0x00ff0000)>>16)/255.f;
            printf("Training: %x, %x, %x\n", img1_pixels[i]&0x000000ff, (img1_pixels[i]&0x0000ff00)>>8, (img1_pixels[i]&0x00ff0000)>>16);
        }
    }
    for (int y = 0; y < img2_height; ++y) {
        for (int x = 0; x < img2_width; ++x) {
            size_t i = img1_width*img1_height + y*img2_width + x;
            size_t j = y*img1_width + x;
            MAT_AT(t, i, 0) = (float)x/(img2_width - 1);
            MAT_AT(t, i, 1) = (float)y/(img2_height - 1);
            MAT_AT(t, i, 2) = 1.f;
            MAT_AT(t, i, 3) = (img2_pixels[j]&0x000000ff)/255.f;
            MAT_AT(t, i, 4) = ((img2_pixels[j]&0x0000ff00)>>8)/255.f;
            MAT_AT(t, i, 5) = ((img2_pixels[j]&0x00ff0000)>>16)/255.f;
        }
    }

    nn_rand(nn, -1, 1);

    size_t WINDOW_FACTOR = 80;
    size_t WINDOW_WIDTH = (16*WINDOW_FACTOR);
    size_t WINDOW_HEIGHT = (9*WINDOW_FACTOR);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "gym");
    SetTargetFPS(60);

    Gym_Plot plot = {0};
    Font font = LoadFontEx("./fonts/iosevka-regular.ttf", 72, NULL, 0);
    SetTextureFilter(font.texture, TEXTURE_FILTER_BILINEAR);

    size_t preview_width = 28;
    size_t preview_height = 28;

    Image preview_image1 = GenImageColor(preview_width, preview_height, BLACK);
    Texture2D preview_texture1 = LoadTextureFromImage(preview_image1);

    Image preview_image2 = GenImageColor(preview_width, preview_height, BLACK);
    Texture2D preview_texture2 = LoadTextureFromImage(preview_image2);

    Image preview_image3 = GenImageColor(preview_width, preview_height, BLACK);
    Texture2D preview_texture3 = LoadTextureFromImage(preview_image3);

    Image original_image1 = GenImageColor(img1_width, img1_height, BLACK);
    for (size_t y = 0; y < (size_t) img1_height; ++y) {
        for (size_t x = 0; x < (size_t) img1_width; ++x) {
            uint32_t pixel = img1_pixels[y*img1_width + x];
            ImageDrawPixel(&original_image1, x, y, CLITERAL(Color) {
                pixel&0x000000ff, (pixel & 0x0000ff00)>>8, (pixel & 0x00ff0000)>>16, 255
            });
        }
    }
    Texture2D original_texture1 = LoadTextureFromImage(original_image1);

    Image original_image2 = GenImageColor(img2_width, img2_height, BLACK);


    for (size_t y = 0; y < (size_t) img2_height; ++y) {
        for (size_t x = 0; x < (size_t) img2_width; ++x) {
            uint32_t pixel = img2_pixels[y*img2_width + x];
            ImageDrawPixel(&original_image2, x, y, CLITERAL(Color) {
                pixel&0x000000ff, (pixel & 0x0000ff00)>>8, (pixel & 0x00ff0000)>>16, 255
            });
        }
    }
    Texture2D original_texture2 = LoadTextureFromImage(original_image2);

    Batch batch = {0};
    bool rate_dragging = false;
    bool scroll_dragging = false;
    size_t epoch = 0;


    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }
        if (IsKeyPressed(KEY_R)) {
            epoch = 0;
            nn_rand(nn, -1, 1);
            plot.count = 0;
        }
        if (IsKeyPressed(KEY_S)) {
            render_upscaled_screenshot(nn, "upscaled.png");
        }

        for (size_t i = 0; i < batches_per_frame && !paused && epoch < max_epoch; ++i) {
            batch_process(&batch, batch_size, nn, g, t, rate);
            if (batch.finished) {
                epoch += 1;
                da_append(&plot, batch.cost);
                mat_shuffle_rows(t);
            }
        }

        MAT_AT(NN_INPUT(nn), 0, 2) = 0.f;
        gym_nn_image_rgb(nn, preview_image1.data, preview_image1.width, preview_image1.height, preview_image1.width, 0, 1);
        UpdateTexture(preview_texture1, preview_image1.data);

        MAT_AT(NN_INPUT(nn), 0, 2) = 1.f;
        gym_nn_image_rgb(nn, preview_image2.data, preview_image2.width, preview_image2.height, preview_image2.width, 0, 1);
        UpdateTexture(preview_texture2, preview_image2.data);

        MAT_AT(NN_INPUT(nn), 0, 2) = scroll;
        gym_nn_image_rgb(nn, preview_image3.data, preview_image3.width, preview_image3.height, preview_image3.width, 0, 1);
        UpdateTexture(preview_texture3, preview_image3.data);

        BeginDrawing();
        Color background_color = {0x18, 0x18, 0x18, 0xFF};
        ClearBackground(background_color);
        {
            int w = GetRenderWidth();
            int h = GetRenderHeight();

            Gym_Rect r;
            r.w = w;
            r.h = h*2/3;
            r.x = 0;
            r.y = h/2 - r.h/2;

            gym_layout_begin(GLO_HORZ, r, 3, 10);
                gym_plot(plot, gym_layout_slot());
                gym_render_nn_weights_heatmap(nn, gym_layout_slot());
                Gym_Rect preview_slot = gym_layout_slot();
                gym_layout_begin(GLO_VERT, preview_slot, 3, 0);
                    gym_layout_begin(GLO_HORZ, gym_layout_slot(), 2, 0);
                        render_texture_in_slot(original_texture1, GHA_RIGHT, GVA_BOTTOM, gym_layout_slot());
                        render_texture_in_slot(original_texture2, GHA_LEFT, GVA_BOTTOM, gym_layout_slot());
                    gym_layout_end();
                    gym_layout_begin(GLO_HORZ, gym_layout_slot(), 2, 0);
                        render_texture_in_slot(preview_texture1, GHA_RIGHT, GVA_TOP, gym_layout_slot());
                        render_texture_in_slot(preview_texture2, GHA_LEFT, GVA_TOP, gym_layout_slot());
                    gym_layout_end();
                    render_texture_in_slot(preview_texture3, GHA_CENTER, GVA_CENTER, gym_layout_slot());
                gym_layout_end();
                {
                    float rw = preview_slot.w;
                    float rh = preview_slot.h*0.03;
                    float rx = preview_slot.x;
                    float ry = rh + preview_slot.y + preview_slot.h;
                    gym_slider(&scroll, &scroll_dragging, rx, ry, rw, rh);
                }
            gym_layout_end();

            char buffer[256];
            snprintf(buffer, sizeof(buffer), "Epoch: %zu/%zu, Rate: %f, Cost: %f", epoch, max_epoch, rate, plot.count > 0 ? plot.items[plot.count - 1] : 0);
            DrawTextEx(font, buffer, CLITERAL(Vector2) {}, h*0.04, 0, WHITE);
            gym_slider(&rate, &rate_dragging, 0, h*0.08, w, h*0.02);
        }
        EndDrawing();
    }

    return 0;

}