import math as Math
import colorsys
import numpy as np
from PIL import Image, ImageFilter
from skimage.color import rgb2lab
import cv2
import copy
import blend_modes
import time
from collections import Counter
from itertools import chain
import os

def main():
    begin = time.time() # Store the start time

    target_image = AutoRough(image_path = 'Images/vijnbjz_2K_Albedo.jpg') # Instantiating an object of class AutoRough, the default values can be replacing by uncommenting.
    # Uncomment and change below default values
                             # number_of_colors = 64)
                             # pixel_count_percentage_threshold = 0.055,
                             # maximum_delta = 20,
                             # minimum_distance = 200,
                             # minimum_occurence = 0.01,
                             # widen_brightness_search_range = True,
                             # regard_saturation = True,
                             # median_filter = True)

    output_masks = target_image.get_masks() # Single function to return masks

    end = time.time() # Store the end time
    print(f"Runtime of the program is {end - begin} seconds")

class AutoRough:

    def __init__(self, iterable=(), **kwargs):
        # Set default values
        self.number_of_colors = 64  # The amount of colors chosen from the start of the complete color palette of the input image.
        # These colors are then quantized and used for segmentation purposes. Default value is 64. A higher number gives higher segmentation accuracy but takes more time.
        self.pixel_count_percentage_threshold = 0.055  # Threshold amount used for separation of colors with high number of pixels. from those with low number of pixels. Default value is 0.055. A higher threshold yields fewer masks, filtered by median reduction.
        self.maximum_delta = 20  # Threshold amount used for comparison of color distances between colors with low pixel counts and colors of low pixel counts. A higher value deletes more residue (does not add low pixel count colors to high pixel count colors).
        self.minimum_distance = 200  # Default distance to be compared with maximum_delta. A lower value will delete more residue.
        self.minimum_occurence = 0.01  # Each color in the final set of quantized_colors has its pixel count compared with this threshold. A higher number yields fewer masks.
        self.widen_brightness_search_range = True  # If true, color variation within the same hue range will be accepted, resulting in more masks but a more accurate albedo representation.
        self.regard_saturation = True  # Increases accuracy in color quantization, resulting in higher number of masks at the cost of more computation time.
        self.median_filter = True  # If true, yields fewer masks.
        self.__dict__.update(iterable, **kwargs)
        self.create_Outputs_folder()    # Create a directory named "outputs" where the masks will be stored

    def dither_image(self):
        """
        Restricts the colors of the image to the specified number, otherwise known
        as dithering. Dithering is done using PIL.Image's convert() method.

        Args:
            None
        Returns:
            Dithered image
        """
        original_image = Image.open(self.image_path)
        dithered_image = original_image.convert(mode='P',
                                         colors=self.number_of_colors,
                                         dither=1,
                                         palette=1)

        return dithered_image

    def deltaEModified(self, color1, color2):
        """
        Aggregates Euclidean distance, RGB distance, and RGB-saturation-brightness distance.
        The formulas for all three distances are as follows:
            > Euclidean distance  √[ (x2– x1)2 + (y2 – y1)2 ]
            > RGB distance = √[(red_color1 - red_color2)^2 + (green_color1 - green_color2)^2 + (blue_color1 - blue_color2)^2]
            > RGB-saturation-brightness distance = [ (rgb_distance / 100 + brightness_distance_factor + saturation_distance_factor) * 100 ] / 2.7321

        Args:
            color1 (list): RGB triplet for the first color
            color2 (list): RGB triplet for the second color
        Returns:
            The calculated distance.
        """
        color1_new = np.divide(np.array(color1), 255.0)
        color2_new = np.divide(np.array(color2), 255.0)

        color1_lab = rgb2lab(color1_new, illuminant="D65", observer="10")
        color2_lab = rgb2lab(color2_new, illuminant="D65", observer="10")

        l1 = color1_lab[0]
        a1 = color1_lab[1]
        b1 = color1_lab[2]
        l2 = color2_lab[0]
        a2 = color2_lab[1]
        b2 = color2_lab[2]

        r1 = color1[0] / 2.55
        r2 = color2[0] / 2.55
        g1 = color1[1] / 2.55
        g2 = color2[1] / 2.55
        bl1 = color1[2] / 2.55
        bl2 = color2[2] / 2.55

        color1_red_percentage = color1[0] / float(255)
        color1_green_percentage = color1[1] / float(255)
        color1_blue_percentage = color1[2] / float(255)
        color1_hsv_percentage = colorsys.rgb_to_hsv(color1_red_percentage, color1_green_percentage,
                                                    color1_blue_percentage)
        h1 = (360 * color1_hsv_percentage[0])
        s1 = (100 * color1_hsv_percentage[1])
        br1 = (100 * color1_hsv_percentage[2])

        color2_red_percentage = color2[0] / float(255)
        color2_green_percentage = color2[1] / float(255)
        color2_blue_percentage = color2[2] / float(255)
        color2_hsv_percentage = colorsys.rgb_to_hsv(color2_red_percentage, color2_green_percentage,
                                                    color2_blue_percentage)
        h2 = (360 * color2_hsv_percentage[0])
        s2 = (100 * color2_hsv_percentage[1])
        br2 = (100 * color2_hsv_percentage[2])

        dE = Math.sqrt(Math.pow(l1 - l2, 2) + Math.pow(a1 - a2, 2) + Math.pow(b1 - b2, 2))
        rgb_distance = Math.sqrt(Math.pow(r1 - r2, 2) + Math.pow(g1 - g2, 2) + Math.pow(bl1 - bl2, 2))

        brightness_distance_factor = abs(br1 - br2) / 100

        br1 = br1 if br1 < 50 else 100 - br1
        br2 = br2 if br2 < 50 else 100 - br2

        brightnessMultiplier = ((br1) / 50) * ((br2) / 50)
        saturation_distance_factor = (abs(s1 - s2) * brightnessMultiplier) / 100
        dERGBSB = (((rgb_distance / 100 + brightness_distance_factor + saturation_distance_factor) * 100) / 2.7321)

        dEAvg = (dE + dERGBSB * 2 + rgb_distance) / 4

        return dEAvg

    def getColorTag(self, color):
        """
        Calculates percentages of the red, green and blue channels. These percentages are used
        by colorsys module's rgb_to_hsv method. The HSV of the color is then used to calculate its
        hue, saturation and brightness. A string is formed that describes the color.

        Args:
            color (list of int): List of values for red, green and blue channels
        Returns:
            A string that specifies the brightness, saturation of hue of the color.
        """
        red_percentage = color[0] / float(255)
        green_percentage = color[1] / float(255)
        blue_percentage = color[2] / float(255)

        # get hsv percentage: range (0-1, 0-1, 0-1)
        color_hsv_percentage = colorsys.rgb_to_hsv(red_percentage, green_percentage, blue_percentage)

        # get normal hsv: range (0-360, 0-255, 0-255)
        hue = round(360 * color_hsv_percentage[0])
        saturation = round(100 * color_hsv_percentage[1])
        brightness = round(100 * color_hsv_percentage[2])

        if (hue < 10): tag = "Red"
        if (hue < 20):
            if (saturation < 50):
                tag = "Brown"
            else:
                tag = "Red-Orange"
        elif (hue < 40):
            if (saturation < 50):
                tag = "Brown"
            else:
                tag = "Orange"
        elif (hue < 50):
            tag = "Orange-Yellow"
        elif (hue < 80):
            tag = "Yellow"
        elif (hue < 169):
            tag = "Green"
        elif (hue < 220):
            tag = "Cyan"
        elif (hue < 280):
            tag = "Blue"
        elif (hue < 330):
            tag = "Magenta"
        else:
            tag = "Red"

        if (brightness < 25):
            tag = "Black-" + tag
        elif (brightness > 75):
            if (saturation < 33): tag = "White-" + tag

        if (self.regard_saturation):
            if (saturation < 10):
                tag = "Gray-" + tag
            elif (saturation < 20):
                tag = "Desaturated-" + tag
            elif (saturation < 40):
                tag = "Medium-Desaturated-" + tag
            elif (saturation > 80):
                tag = "Highly-Saturated-" + tag
            elif (saturation > 60):
                tag = "Saturated-" + tag
        elif (saturation < 20):
            tag = "Gray"

        if (self.widen_brightness_search_range):
            if (brightness < 30):
                tag = "Dark-" + tag
            elif (brightness < 40):
                tag = "Medium-Dark-" + tag
            elif (brightness > 70):
                tag = "Light-" + tag
            elif (brightness > 60):
                tag = "Medium-Light-" + tag
        else:
            if (brightness < 30):
                tag = "Dark-" + tag
            elif (brightness > 60):
                tag = "Light-" + tag

        return tag

    def sort_colors_by_pixel_count(self, colors):
        """
        Sorts the list containing RGB triplets and their pixel counts based on the pixel counts
        in ascending fashion.

        Args:
            colors (list of lists of (list of int, str, int))): Each sublist contains the RGB triplet, tag and pixel count of single color.
        Returns:
            Sorted list.
        """
        for k in range(len(colors)):
            pixel_count = colors[k][2]
            color = colors[k]
            l = k - 1
            while l >= 0 and colors[l][2] > pixel_count:
                colors[l + 1] = colors[l]
                l = l - 1
            colors[l + 1] = color
        return colors

    def GIF2RGB(self, gif):
        """
        Converts GIF image to list containing RGB triplets.
        Args:
            gif (PIL object): The gif image opened by PIL.
        Returns:
            RGB array.
        """
        rgb_image = gif.convert('RGB')
        rgb_arr = np.array(rgb_image.getdata()).reshape(rgb_image.size[0], rgb_image.size[1], 3)

        return rgb_arr

    def flatten(self, xss):
        """
        Converts list of lists into a single list.

        Args:
            xss (list of lists):
        Returns:
            Flattened list.
        """
        return [x for xs in xss for x in xs]

    def average_colors_of_each_tag(self, quantized_colors, tagged_colors, tags):
        """
        Averages all colors of each tag, then uses average to update quantized_colors and tagged_colors lists.
        Colors belonging to the same tag have their RGB values averaged, scaled by a weight and summed.
        An Average Color is formed using the summed averaged RGB values, which replaces colors of its tag in
        quantized_colors and tagged_colors. Average Colors are also appended into a unique_colors list.

        Args:
            quantized_colors (list of lists)
            tagged_colors (list of lists)
            tags (list): Unique tags
        Returns:
            Updated quantized_colors and tagged_colors lists.
        """
        unique_colors = []
        for i in range(len(tags)):
            tag = tags[i]
            colors = []

            for j in range(self.number_of_colors):
                if (tag == tagged_colors[j][1]):
                    colors.append(tagged_colors[j])

            colors = self.sort_colors_by_pixel_count(colors)

            avg_color = [0, 0, 0]
            avgR = 0
            avgG = 0
            avgB = 0
            max_pixel_count = colors[len(colors) - 1][2]
            weightsum = 0

            for j in range(len(colors)):
                weight = colors[j][2] / max_pixel_count
                rgb = colors[j][0]

                avgR += rgb[0] * weight
                avgG += rgb[1] * weight
                avgB += rgb[2] * weight
                weightsum += weight

            avg_color[0] = min(avgR / weightsum, 255)
            avg_color[1] = min(avgG / weightsum, 255)
            avg_color[2] = min(avgB / weightsum, 255)
            colors[len(colors) - 1][0] = avg_color

            for j in range(len(colors) - 1):
                colors[len(colors) - 1][2] += colors[j][2]

            for j in range(self.number_of_colors):
                if (tag == tagged_colors[j][1]):
                    quantized_colors[j] = colors[len(colors) - 1][0]
                    tagged_colors[j] = colors[len(colors) - 1]

            unique_colors.append(colors[len(colors) - 1])

        return unique_colors, quantized_colors, tagged_colors

    def get_unique_tags_and_pixel_counts(self, palette, pixel_counts):
        """
        The first (number_of_colors) colors from the color palette of the image have their tags set
        and pixel counts obtained from the pixel_counts dictionary. Unique tags are placed
        in a 'tags' list. A tagged_colors list in defined composed of RGB triplets, their
        tags and pixel counts.

        Args:
            palette (list): unrestricted color palette
            pixel_counts (dictionary): key-value pairs of colors and their pixel counts
        Returns:
            List of unique tags, list of colors and their tags and pixel counts
        """
        tags = []
        tagged_colors = []

        for i in range(self.number_of_colors):
            col = palette[i]
            tag = self.getColorTag(col)
            tagExists = False

            pixel_count = pixel_counts[col]

            for j in range(len(tags)):
                if tags[j] == tag:
                    tagExists = True

            if not tagExists:
                tags.append(tag)

            tagged_colors.append([col, tag, pixel_count])

        return tags, tagged_colors

    def get_color_palette(self, image):
        """
        Uses PIL.Image's getpalette() method to obtain color palette of image, which is a list
        containing the RGB triplets of the colors in the image. Returns one list containing
        all the colors from the image, and another list containing 'number_of_colors' amount of colors.

        Args:
            image (PIL object): the image read using Image.open
        Returns:
            A list containing the complete color palette of input image, and another list containing colors equal to number_of_colors.
        """
        im2 = image.getpalette()
        complete_palette = [tuple(im2[i:i + 3]) for i in range(0, len(im2), 3)]
        restricted_palette = complete_palette[:self.number_of_colors]

        return complete_palette, restricted_palette

    def pixels_per_color(self, gifimage):
        """
        Returns a count of pixels per unique color (unique RGB combination. Input GIF image is converted to an np array,
        which is transformed from being a list of listsf of lists to a list of tuples. Itertools.chain and
        Counter are used to count number of occurence of each tuple in the list. A dictionary is
        created holding the RGB triplets and their counts.

        Args:
            gifimage (str): the image to count the number of pixels of
        Returns:
            A key-value pairing of the rgb color value and the number of times the color was present in the image
        """
        width, height = gifimage.size
        rgb_image = gifimage.convert('RGB')
        rgb_array = np.array(rgb_image.getdata()).reshape(rgb_image.size[0], rgb_image.size[1], 3)

        # Convert rgb_array into list of tuples and create Counter object to count occurrence of unique tuples
        res = [[tuple(subsublist) for subsublist in sublist] for sublist in rgb_array]
        color_counts = Counter(chain(*res))

        return gifimage, color_counts, width, height, rgb_array

    def separate_colors_with_high_pixel_count(self, unique_colors, tagged_colors, quantized_colors, aw, ah):
        """
        Separates colors with high pixel count into filtered colors and the rest in residue colors.
        Calculates Modified Delta-E distance of each residue color from all filtered colors. The filtered
        color from which DEMod is lowest, has the residue color's RGB values scaled by weight and added
        to it. quantized_colors and tagged_colors are updated using the filtered colors.

        Args:
            unique_colors (list of lists of (list of float, str, int))
            tagged_colors (list of lists of (list of float, str, int))
            quantized_colors (list of int)
            aw (int)
            ah (int)
        Returns:
            Updated quantized_colors and tagged_colors.
        """
        filtered_colors = []
        color_residue = []
        sortedunique_colors = self.sort_colors_by_pixel_count(unique_colors)
        sortedunique_colors.reverse()

        for i in range(len(sortedunique_colors)):
            if (sortedunique_colors[i][2] / (aw * ah)) > self.pixel_count_percentage_threshold:
                filtered_colors.append(sortedunique_colors[i])
            else:
                color_residue.append(sortedunique_colors[i])

        for i in range(len(color_residue)):
            index_match = -1
            dEMin = self.minimum_distance

            for j in range(len(filtered_colors)):
                dEMod = self.deltaEModified(color_residue[i][0], filtered_colors[j][0])

                if dEMod < dEMin:
                    dEMin = dEMod
                    index_match = j

            if dEMin < self.maximum_delta:
                for j in range(self.number_of_colors):
                    if tagged_colors[j][1] == color_residue[i][1]:
                        weight = color_residue[i][2] / filtered_colors[index_match][2]
                        filtered_colors[index_match][0][0] = (filtered_colors[index_match][0][0] + color_residue[i][0][
                            0] * weight) / (1 + weight)
                        filtered_colors[index_match][0][1] = (filtered_colors[index_match][0][1] + color_residue[i][0][
                            1] * weight) / (1 + weight)
                        filtered_colors[index_match][0][2] = (filtered_colors[index_match][0][2] + color_residue[i][0][
                            2] * weight) / (1 + weight)

                        filtered_colors[index_match][2] += color_residue[i][2]
                        tagged_colors[j] = filtered_colors[index_match]
                        quantized_colors[j] = filtered_colors[index_match][0]

        return tagged_colors, quantized_colors

    def black_and_white_masks(self, unique_colors, quantized_colors, gif_image, aw, ah):
        """
        Creates black and white mask for each unique color. Uses PIL.Image's putpalette method
        to put quantized_colors as the color palette of original image to create new image.
        Rounds all unique colors. For each unique color, create a (aw x ah) np array full of
        the color, then compare it with array of new image to place 1's at True locations.
        These 1's are changed into 255's, resulting in white pixels at locations where the
        color is present in the new image, and black and white mask is formed. Applies median
        filtering of size max(aw, ah)/512 + 1. Now if the white pixels are greater than minimum
        occurence, the color is appended into unique_colorsMerged and the black and white mask
        is saved.

        Args:
            unique_colors (list of lists of (list of float, str, int))
            quantized_colors (list of int)
            gif_image (PIL Image object)
            aw (int)
            ah (int)
        Returns:
            A list with unique colors having high occurence in image, and a list containing black and white masks.
        """
        unique_colors = self.sort_colors_by_pixel_count(unique_colors)
        unique_colors.reverse()
        unique_colorsMerged = []

        rounded_palette = self.flatten(np.around(quantized_colors).astype(np.uint8))
        gif_image.putpalette(rounded_palette)
        gif_image.save('Reduced Color Image.png')
        conv_image = self.GIF2RGB(gif_image)
        conv_array = np.array(conv_image)
        masks = []

        rounded_colors = [np.around(item[0]) for item in unique_colors]  # Round all rgb values in unique_colors list
        for newValue, subList in zip(rounded_colors, unique_colors):
            subList[0] = [int(value) for value in newValue]

        for i in range(len(unique_colors)):
            color_array = np.full([aw, ah, 3], unique_colors[i][0])
            blackwhite = (conv_array == color_array)
            blackwhite = np.uint8(blackwhite)
            blackwhite[(blackwhite == 1).all(axis=-1)] = (255, 255, 255)
            blackwhite[(blackwhite == 0).any(axis=-1)] = (0, 0, 0)

            if self.median_filter is True:
                kernel = int((max(aw, ah) / 512) + 1)
                filtered_blackwhite = cv2.medianBlur(blackwhite, kernel)
                pixels = np.sum(filtered_blackwhite == 255) / 3
            else:
                pixels = np.sum(blackwhite == 255) / 3

            occurence = pixels / (aw * ah)
            if occurence > self.minimum_occurence:
                unique_colorsMerged.append(unique_colors[i])
                masks.append([unique_colors[i][1], blackwhite])

        for i in range(len(unique_colorsMerged)):
            print("Color #", i + 1, ": ", unique_colorsMerged[i])

        return unique_colorsMerged, masks

    def colorize_masks_and_merge_masks(self, unique_colors, masks):
        """
        Replaces white pixels with color pixels. Adds alpha channel to this
        mask. Reduces opaciy of black pixels to 0, making them transparent.
        Photoshop uses Refine Edge to refine the colorized masks. This feature
        is not implemented in Python yet. Resultant masks are pasted upon each
        other using PIL.Image's paste() method.

        Args:
            unique_colors (list of lists of (list of float, str, int)):
            masks (list of lists of (str, list of lists of int):
        Returns:
            PIL image object of colorized masks pasted upon each other, and list of PIL Image objects
        """
        list_of_masks = []
        dictionary_of_masks = {}
        for i in range(len(unique_colors)):
            mask = self.white_pixels_to_color_pixels(masks[i][1], i + 1, unique_colors[i][0], masks[i][0])
            list_of_masks.append(mask)
            dictionary_of_masks[masks[i][0]] = mask
            if i == 0:
                masks_pasted = mask.copy()
            if i > 0:
                masks_pasted.paste(mask, (0, 0), mask)

        return masks_pasted, list_of_masks, dictionary_of_masks

    def create_base_layer(self, masks_pasted, rgb_array, aw, ah):
        """
        Creates base color layer. Masks pasted upon each other are subtracted from image
        to get unmasked regions. This is done by setting all pixels having RGB greater than
        0 to 255. After subtraction, all values below 0 are set to 0. Average RGB triplet of the
        unmasked array, pasted massk array and image array is calculated using np.average.
        This average color replaces all color pixels in unmasked regions image. The unmasked array,
        masks array and image array are merged using np.sum. Now base color is calculated from this
        merged image. The first intensity level at each channel containing pixels makes up the base
        color. Base layer is created full of base color pixels.

        Args:
            masks_pasted (PIL Image object)
            rgb_array (list of lists of int)
            aw (int)
            ah (int)
        Returns:
            PIL image of the base color layer.
        """
        # Subtract masks from original image to get unmasked areas
        masks_array = np.array(masks_pasted)[:, :, :3]
        new_summed_masks = copy.deepcopy(masks_array)
        new_summed_masks[(new_summed_masks > 0).all(axis=-1)] = (255, 255, 255)
        unmasked = np.subtract(rgb_array, new_summed_masks)
        unmasked[(unmasked < 0).all(axis=-1)] = (0, 0, 0)
        unmasked = unmasked.astype(np.uint8)

        # Put average color of all layers in unmasked image
        add_all = [unmasked, masks_array, rgb_array]
        average_color = np.average(np.average(np.average(add_all, axis=0), axis=0), axis=0)
        unmasked[(unmasked != 0).all(axis=-1)] = average_color
        new_unmasked = copy.deepcopy(unmasked)
        new_unmasked[(new_unmasked > 0).all(axis=-1)] = (255, 255, 255)

        # Merge original image with summed masks and average color layer
        new_rgb_array = np.subtract(np.subtract(rgb_array, new_summed_masks), new_unmasked)
        new_rgb_array[(new_rgb_array < 0).all(axis=-1)] = (0, 0, 0)
        new_rgb_array = np.array([new_rgb_array, masks_array, unmasked]).sum(axis=0)
        new_rgb_array = new_rgb_array.astype(np.uint8)

        baseColor = []

        bgr = cv2.split(new_rgb_array)
        for chan in bgr:
            # create a histogram for the current channel and plot it
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            for i in range(len(hist)):
                if (hist[i]) != 0:
                    baseColor.append(i)
                    break

        print("Base color: ", baseColor)
        base_color = np.full((aw, ah, 3), baseColor)
        base_color_layer = Image.fromarray(base_color.astype(np.uint8))

        return base_color_layer

    def create_detail_layer(self, list_of_masks, base_color_layer, rgb_array):
        """
        Creates detail layer. Pastes all colorized masks onto base color layer using
        PIL.Image's paste() method. Inverts this image using cv2 and adds alpha channel
        to it and the array of original image. Photoshop uses linear light blending, in Python
        hard light blending is used for similar result. Method hard_light() in blend_modes
        module is used to blend the two images. Resulting image is detail layer.
        Args:
            list_of_masks (list of lists)
            base_color_layer (PIL image object)
            rgb_array (list of lists of int)
        """
        for i in list_of_masks:
            base_color_layer.paste(i, (0, 0), i)

        # Invert detail layer
        inverted = cv2.bitwise_not(np.array(base_color_layer))
        detail = np.empty((inverted.shape[0], inverted.shape[1], 4)).astype("float32")
        detail[:, :, 0:3] = inverted
        detail[:, :, 3] = 255

        # Add alpha channel to background layer
        bg = np.empty((rgb_array.shape[0], rgb_array.shape[1], 4)).astype("float32")
        bg[:, :, 0:3] = rgb_array
        bg[:, :, 3] = 255

        detail_layer = np.uint8(blend_modes.hard_light(bg, detail, 0.5))
        detail_layer_image = Image.fromarray(detail_layer.astype(np.uint8))

        return detail_layer_image

    def white_pixels_to_color_pixels(self, mask, number, color, tag):
        """
        Colorizes the black and white masks.
        Note: Work in progress. This function does not mimic Photoshop's Refine Edge feature.
        It first gives color to white pixels. Then, it dilates the mask to get dilations,
        takes the morphological gradient to get the gradient, then reduces their opacities.
        The black pixels are made transparent.
        Args:
            mask (list):
            number (int):
            color (list):
            tag (str):
        Returns:
            Colorized mask.
        """
        img = Image.fromarray(mask.astype(np.uint8))
        img = img.filter(ImageFilter.ModeFilter(size=2))

        # # Binary colored mask
        # mask[(mask == 255).all(axis=-1)] = color
        # mask_original = mask

        # Dilate image
        mask_undilated = np.array(img)
        mask_dilated = cv2.dilate(mask_undilated, np.ones((2, 2), np.uint8), iterations=1)
        dilations = mask_dilated - mask_undilated

        gradient = cv2.morphologyEx(mask_undilated, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))

        mask = mask_undilated - gradient

        # Change black and white mask to colored mask
        mask[(mask == 255).all(axis=-1)] = color
        gradient[(gradient == 255).all(axis=-1)] = (1, 1, 1)
        dilations[(dilations == 255).all(axis=-1)] = (2, 2, 2)
        array_list = np.array([mask, gradient, dilations])
        mask = array_list.sum(axis=0)

        # Create transparent image
        # Add Alpha (Opacity) channel to colored mask
        temp = np.empty((mask.shape[0], mask.shape[1], 4))
        temp[:, :, 0:3] = mask
        temp[:, :, 3] = 255

        img = Image.fromarray(temp.astype(np.uint8))
        datas = img.getdata()

        # Change Alpha channel of black pixels to zero
        newData = []
        for item in datas:
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                newData.append((0, 0, 0, 0))
            elif item[0] == 1 and item[1] == 1 and item[2] == 1:
                newData.append((int(color[0]), int(color[1]), int(color[2]), 245))
            elif item[0] == 2 and item[1] == 2 and item[2] == 2:
                newData.append((int(color[0]), int(color[1]), int(color[2]), 220))
            elif item[0] == 4 and item[1] == 4 and item[2] == 4:
                newData.append((int(color[0]), int(color[1]), int(color[2]), 200))
            else:
                newData.append(item)

        # Save new mask
        img.putdata(newData)
        name = "Outputs/Mask #" + str(number) + " (" + str(tag) + ").png"
        print(name, "creation successful.")

        return img

    def get_masks(self):
        """
        Runs all required methods to produce the masks. The sequence of methods is as follows:
        1. Dither input image.
        2. Create dictionary of pixel counts of each unique color.
        3. Obtain color palette.
        4. Obtain unique tags and pixel counts of unique colors.
        5. Average all colors belonging to same tags.
        6. Separate colors with high pixel counts, add low pixel count colors to matching high pixel count ones
        7. Create black and white masks.
        8. Colorize the masks.
        9. Create base layer and detail layer.
        Args:
            None
        Returns:
            Dictionary of colorized and refined masks
        """
        # Dither the image. Reduce color palette and apply error-diffusion dithering across the image.
        dithered_image = self.dither_image()

        # Create dictionary of pixel count of all unique colors in dithered image
        gif_image, color_counts, aw, ah, rgb_array = self.pixels_per_color(dithered_image)
        print("Dimensions: ", aw, "x", ah, "\n")

        # Color Quantization
        # STAGE 1: Get color palette
        quantized_colors, palette = self.get_color_palette(gif_image)

        # STAGE 2: Get tag and pixel count of each color
        tags, tagged_colors = self.get_unique_tags_and_pixel_counts(palette, color_counts)

        # STAGE 3: For each tag, average colors of that tag
        unique_colors, quantized_colors, tagged_colors = self.average_colors_of_each_tag(quantized_colors,
                                                                                              tagged_colors, tags)

        # STAGE 4: Separate colors with high pixel count
        tagged_colors, quantized_colors = self.separate_colors_with_high_pixel_count(unique_colors, tagged_colors,
                                                                                          quantized_colors, aw, ah)

        # Layers Creation
        # STAGE 1: Create black and white masks of colors with high occurence
        unique_colors, masks = self.black_and_white_masks(unique_colors, quantized_colors, gif_image, aw, ah)

        # Stage 2: Create colorized masks
        masks_pasted, list_of_masks, dictionary_of_masks = self.colorize_masks_and_merge_masks(unique_colors, masks)

        # Stage 3: Create Base Color layer
        base_color_layer = self.create_base_layer(masks_pasted, rgb_array, aw, ah)

        # Stage 4: Create Detail layer
        detail_layer = self.create_detail_layer(list_of_masks, base_color_layer, rgb_array)

        # Update dictionary of masks
        dictionary_of_masks['Base Layer'] = base_color_layer
        dictionary_of_masks['Detail Layer'] = detail_layer

        return list_of_masks

    def create_Outputs_folder(self):
        """
        Creates directory "Outputs/" in root directory of project.
        """
        parent = os.getcwd()
        directory = "Outputs"
        path = os.path.join(parent, directory)

        if os.path.isdir(path):
            return
        os.mkdir(path)

if __name__ == '__main__':
    main()