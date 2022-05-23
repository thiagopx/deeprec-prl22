import os
import re
import numpy as np
import cv2
import random
import copy
from ndarray import first_nonzero, last_nonzero
from collections import defaultdict
from alignment import icp_least_squares as icp, best_rotation, best_translation

# from .alignment import run_cpd2d
from ndarray import first_nonzero, last_nonzero
from utils import grayscale_to_rgb


def shred(
    image, num_strips, index=None, filter_blanks=False
):  # index is not used for now
    """Artifical shredding."""

    x = 0
    strips_list = []
    for index in range(num_strips):
        dw = int((image.shape[1] - x) / (num_strips - index))
        crop = image[:, x : x + dw]
        strip = Strip(crop, index=index, position=(x, 0))  # artificial mask is True
        strips_list.append(strip)
        x += dw
    strips = Strips(strips_list=strips_list, filter_blanks=filter_blanks)
    return strips


def hdist(border1, border2):
    """Horizontal distance."""

    border1_dict = defaultdict(int)
    for y, x in zip(border1[:, 1], border1[:, 0]):
        border1_dict[y] = max(border1_dict[y], x)

    border2_dict = defaultdict(lambda: float("inf"))
    for y, x in zip(border2[:, 1], border2[:, 0]):
        border2_dict[y] = min(border2_dict[y], x)

    min_dist = float("inf")
    touching_point = None
    for y, x in border1_dict.items():
        # min_dist = min(min_dist, border2_dict[y] - x)
        if border2_dict[y] - x < min_dist:
            min_dist = border2_dict[y] - x
            touching_point = (x, y, 1)

    return min_dist, touching_point


def center(border):
    """Central point excluding repeated x/y-values."""

    x = np.unique(border[:, 0]).mean()
    y = np.unique(border[:, 1]).mean()
    return np.array([x, y, 1], dtype=np.int0)


class Strip(object):
    """Strip image."""

    def __init__(
        self, image, index, mask=None, perc_discard_borders=0.01, position=(0, 0)
    ):

        # 2d global position of the top-left corner of the image
        #
        # homegenous coordinates
        self.position = np.array([position[0], position[1], 1], dtype=np.int0)
        # simple form
        self.x = position[0]
        self.y = position[1]

        # image dimensions
        h, w = image.shape[:2]
        self.h = h
        self.w = w

        # mask is stored as uint8
        if mask is not None:
            assert mask.dtype == np.uint8
            self.mask = mask
            self.artificial_mask = False
        else:
            self.mask = 255 * np.ones((h, w), dtype=np.uint8)
            self.artificial_mask = True

        # image content
        assert image.ndim in [2, 3]
        assert image.dtype == np.uint8
        if image.ndim == 2:
            image = grayscale_to_rgb(image)
        self.image = cv2.bitwise_and(image, image, mask=mask).copy()

        # ground-thruth position
        self.index = index

        # left and right borders (opencv representation)
        # - local image space/coordinates
        # - arrays of np.int0 (opencv compatibility)
        self.perc_discard_borders = perc_discard_borders
        self.borders = dict(left=None, right=None)

        # first and last pixel in each row
        self.extremities = dict(left=None, right=None)
        self._detect_borders_extremities()

    def bbox(self):
        """Return the bounding box of an artifial strip."""

        return self.x, self.y, self.w, self.h

    def copy(self):
        """Copy object."""

        return copy.deepcopy(self)

    def _detect_borders_extremities(self):
        """Detect borders/countours of the strip."""

        # 1) borders (or contours)
        contours, _ = cv2.findContours(
            self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cnt = contours[0]
        for cnt_ in contours:
            if cnt_.shape[0] > cnt.shape[0]:
                cnt = cnt_
        cnt = np.squeeze(cnt, axis=1)
        # print(cnt.shape)

        # split contour
        y_min = max(
            int(self.perc_discard_borders * self.h), 1
        )  # always ignore the first row
        y_max = min(
            int((1 - self.perc_discard_borders) * self.h - 1), self.h - 2
        )  # always ignore the last row

        # find range of the left boundary
        idx = 0
        while cnt[idx, 1] < y_min:
            idx += 1

        start_left = idx
        while cnt[idx, 1] <= y_max:
            idx += 1
        end_left = idx

        # find range of the right boundary
        while cnt[idx, 1] > y_max:
            idx += 1
        start_right = idx
        while cnt[idx, 1] > y_min:
            idx += 1
        end_right = idx

        self.borders["left"] = cnt[start_left : end_left + 1]
        self.borders["right"] = cnt[start_right : end_right + 1][::-1]  # reverse

        # homogeneous_coordinates:
        self.borders["left"] = np.pad(
            self.borders["left"], [(0, 0), (0, 1)], mode="constant", constant_values=1
        ).astype(np.int0)
        self.borders["right"] = np.pad(
            self.borders["right"], [(0, 0), (0, 1)], mode="constant", constant_values=1
        ).astype(np.int0)

        # 2) extremities
        self.extremities["left"] = np.apply_along_axis(first_nonzero, 1, self.mask)
        self.extremities["right"] = np.apply_along_axis(last_nonzero, 1, self.mask)

    def rectify(self, border_side="right"):
        assert border_side in ["left", "right"]

        image = np.zeros_like(self.image)
        mask = np.zeros_like(self.mask)

        # right extremity (min)
        border = self.extremities[border_side]
        if border_side == "right":
            for y in range(image.shape[0]):
                dx = image.shape[1] - border[y] - 1
                image[y, dx:] = self.image[y, : border[y] + 1]
                mask[y, dx:] = self.mask[y, : border[y] + 1]
        else:
            for y in range(image.shape[0]):
                dx = border[y]
                image[y, : image.shape[1] - dx] = self.image[y, dx:]
                mask[y, : image.shape[1] - dx] = self.mask[y, dx:]

        # border = self.borders["right"]
        # y_border_min = border[:, 1].min()
        # y_border_max = border[:, 1].max()
        # border_dict = defaultdict(int)
        # for y, x in zip(border[:, 1], border[:, 0]):
        #     border_dict[y] = max(border_dict[y], x)

        # # [0, y_border_min)
        # dx = image.shape[1] - border_dict[y_border_min] - 1
        # image[:y_border_min, dx:] = self.image[:y_border_min, : border_dict[y_border_min] + 1]
        # mask[:y_border_min, dx:] = self.mask[:y_border_min, : border_dict[y_border_min] + 1]

        # # (y_border_max, h - 1]
        # dx = image.shape[1] - border_dict[y_border_max] - 1
        # image[y_border_max + 1 :, dx:] = self.image[y_border_max + 1 :, : border_dict[y_border_max] + 1]
        # mask[y_border_max + 1 :, dx:] = self.mask[y_border_max + 1 :, : border_dict[y_border_max] + 1]

        # for y in range(y_border_min, y_border_max + 1):
        #     dx = image.shape[1] - border_dict[y] - 1
        #     image[y, dx:] = self.image[y, : border_dict[y] + 1]
        #     mask[y, dx:] = self.mask[y, : border_dict[y] + 1]

        self.image = image
        self.mask = mask  # .copy()
        self._detect_borders_extremities()

        return self

    def warp_affine(self, transformation):

        # translate to the global space according the position (x, y)
        W_glob = np.array(
            [
                [1.0, 0.0, self.position[0]],
                [0.0, 1.0, self.position[1]],
                [0.0, 0.0, 1.0],
            ]
        )
        W = transformation @ W_glob

        # corners in homogeneous coordinates (image space)
        #
        # bounding box
        proj_v, proj_h = np.any(self.mask.astype(np.bool), axis=0), np.any(
            self.mask.astype(np.bool), axis=1
        )
        x_min, x_max = first_nonzero(proj_v), last_nonzero(proj_v)
        y_min, y_max = first_nonzero(proj_h), last_nonzero(proj_h)
        # corners coordinates
        corners = np.array(
            [
                [x_min, y_min, 1],  # top-left
                [x_max, y_min, 1],  # top-right
                [x_min, y_max, 1],  # bottom-left
                [x_max, y_max, 1],  # bottom-right
            ]
        )
        # corners coodiantes and bounding box after transformation
        corners_transf = corners @ W.T
        x_trans_min, x_trans_max = int(corners_transf[:, 0].min()), int(
            corners_transf[:, 0].max() + 0.5
        )
        y_trans_min, y_trans_max = int(corners_transf[:, 1].min()), int(
            corners_transf[:, 1].max() + 0.5
        )
        w_trans = x_trans_max - x_trans_min + 1
        h_trans = y_trans_max - y_trans_min + 1

        # warping
        #
        # adjusting transformation: global -> local space (new bounding box)
        W_loc = np.array(
            [[1.0, 0.0, -x_trans_min], [0.0, 1.0, -y_trans_min], [0.0, 0.0, 1.0]]
        )
        W_img = W_loc @ W
        # warping image
        warped = cv2.warpAffine(self.image, W_img[:-1], dsize=(w_trans, h_trans))
        # warped = np.clip(warped, 0, 255).astype(np.uint8)
        # warping mask
        warped_mask = cv2.warpAffine(self.mask, W_img[:-1], dsize=(w_trans, h_trans))
        warped_mask = (255 * (warped_mask == 255)).astype(np.uint8)

        # trimming
        #
        # bounding box
        proj_v, proj_h = np.any(warped_mask.astype(np.bool), axis=0), np.any(
            warped_mask.astype(np.bool), axis=1
        )
        x_trim_min, x_trim_max = first_nonzero(proj_v), last_nonzero(proj_v)
        y_trim_min, y_trim_max = first_nonzero(proj_h), last_nonzero(proj_h)
        # triming image and mask
        trimmed = warped[y_trim_min : y_trim_max + 1, x_trim_min : x_trim_max + 1]
        trimmed_mask = warped_mask[
            y_trim_min : y_trim_max + 1, x_trim_min : x_trim_max + 1
        ]
        # adjust position
        position = np.array(
            [x_trans_min + x_trim_min, y_trans_min + y_trim_min, 1], dtype=np.int0
        )

        # warping borders keeping them at local space (trimmed image)
        W_loc = np.array(
            [[1.0, 0.0, -position[0]], [0.0, 1.0, -position[1]], [0.0, 0.0, 1.0]]
        )
        W_bor = W_loc @ W
        self.borders["left"] = (self.borders["left"] @ W_bor.T).astype(np.int0)
        self.borders["right"] = (self.borders["right"] @ W_bor.T).astype(np.int0)

        # update object state
        self.position = position
        self.image = trimmed
        self.mask = trimmed_mask
        self.h, self.w = trimmed_mask.shape
        return self

    def vert_shift(self, disp):
        """Shift strip vertically."""

        # transformation
        W_disp = np.array([[1.0, 0.0, 0], [0.0, 1.0, disp], [0.0, 0.0, 1.0]])
        self.warp_affine(W_disp)
        return self

    def filled_image(self):
        """Return image with masked-out areas in white."""

        return cv2.bitwise_or(
            self.image, cv2.cvtColor(cv2.bitwise_not(self.mask), cv2.COLOR_GRAY2RGB)
        )

    def is_blank(self, blank_tresh=127):
        """Check whether is a blank strip."""

        blurred = cv2.GaussianBlur(
            cv2.cvtColor(self.filled_image(), cv2.COLOR_RGB2GRAY), (5, 5), 0
        )
        return (blurred < blank_tresh).sum() == 0

    def gap_area(self):

        x_min = self.borders["left"][:, 0].max()
        x_max = self.borders["right"][:, 0].min()
        y_min = min(self.borders["left"][:, 1].min(), self.borders["right"][:, 1].min())
        y_max = min(self.borders["left"][:, 1].max(), self.borders["right"][:, 1].max())
        gap = 255 - self.mask
        gap[:, :x_min] = 0
        gap[y_min:, x_max:] = 0
        gap[:y_min] = 0
        gap[y_max:] = 0
        # for y, (l, r) in enumerate(zip(self.offsets_l, self.offsets_r)):
        #     gap[y, : l] = 0
        #     gap[y, r : ] = 0
        return gap

    def correct_rotation(self, max_theta=5.0, num_angles_per_degree=1, sample_factor=1):
        """Rotate the strip so that left border is laigned with the left border of the image frame."""

        # left border of the...
        #
        # strip
        points_src = self.borders["left"] + (self.position[0], self.position[1], 0)
        # image frame
        points_tgt = np.array(
            [
                self.position[0] * np.ones(self.h),  # x
                np.arange(self.position[1], self.position[1] + self.h),  # y
                np.ones(self.h),
            ],  # homog. coordinate
            dtype=np.int0,
        ).T

        W = best_rotation(
            points_src[::sample_factor],
            points_tgt[::sample_factor],
            max_theta=max_theta,
            num_angles_per_degree=num_angles_per_degree,
            center=self.position,
        )
        self.warp_affine(W)

    def hstack(
        self,
        other,
        pre_alignment=None,
        fine_alignment=None,
        iterations=10,
        sample_factor=1,
        threshold=200,  # icp parameters
        max_theta=5,
        num_angles_per_degree=1,  # best rot parameters
        max_dx=3,
        max_dy=3,  # best trans parameters
    ):
        """Stack horizontally with other strip."""

        assert not (pre_alignment is None and fine_alignment is not None)

        # transformations
        W_origin = np.array(
            [
                [1.0, 0.0, -other.x + self.x],
                [0.0, 1.0, -other.y + self.y],
                [0.0, 0.0, 1.0],
            ]
        )
        W_coarse = np.eye(3)
        W_fine = np.eye(3)

        # no aligment
        if pre_alignment == fine_alignment == None:
            # shift horizontally the other strip to end of the self
            W_coarse = np.array([[1.0, 0.0, self.w], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # pre (coarse) alignment + fine alingment (optional)
        else:
            assert pre_alignment in ["trivial", "centroid"]

            # the alignment will produce at least one 'touching point' that can be used for further fine alignment.
            touching_point = None

            # transformation: local -> global space
            #
            # self right borders (target)
            points_tgt = self.borders["right"] + (self.position[0], self.position[1], 0)
            # other left border (source)
            points_src = other.borders["left"] + (
                other.position[0],
                other.position[1],
                0,
            )

            if pre_alignment == "trivial":
                # pre (coarse) alignment by maximal non-overlapping shift
                temp = points_src + (points_tgt[:, 0].max(), 0, 0)
                dist, touching_point = hdist(points_tgt, temp)
                shift = (points_tgt[:, 0].max() - dist, 0, 0)
            else:
                # pre (coarse) alignment by centroid shift
                # shift = (points_tgt.mean(axis=0) - points_src.mean(axis=0)).astype(np.int0)
                # shift= [0,0]
                shift = center(points_tgt) - center(points_src)
                touching_point = center(points_tgt)
                # shift[1] = (np.unique(points_tgt[:, 1]).mean() - np.unique(points_src[:, 1]).mean()).astype(np.int0)
                # shift = (points_tgt.mean(axis=0) - points_src.mean(axis=0)).astype(np.int0)
            W_coarse = np.array(
                [[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]], [0.0, 0.0, 1.0]]
            )

            # fine alignment
            if fine_alignment is not None:
                assert fine_alignment in ["best-rot", "best-trans", "icp"]

                points_src = points_src @ W_coarse.T
                # fine alignment by searching best rotation
                if fine_alignment == "best-rot":
                    assert touching_point is not None
                    W_fine = best_rotation(
                        points_src[::sample_factor],
                        points_tgt[::sample_factor],
                        max_theta=max_theta,
                        num_angles_per_degree=num_angles_per_degree,
                        center=touching_point,
                    )
                # fine alignment by searching best translation
                elif fine_alignment == "best-trans":
                    W_fine = best_translation(
                        points_src[::sample_factor],
                        points_tgt[::sample_factor],
                        max_dx=max_dx,
                        max_dy=max_dy,
                    )
                # icp
                else:
                    W_fine, _ = icp(
                        points_src[::sample_factor, :-1],
                        points_tgt[::sample_factor, :-1],
                        iterations=iterations,
                        kernel=lambda error: 1.0
                        if np.linalg.norm(error) < threshold
                        else 0.0,
                    )

        temp = other.position
        other = other.copy().warp_affine(W_fine @ W_coarse @ W_origin)
        # parameters of the 'union' between self and other
        x_union_min = min(self.position[0], other.position[0])
        x_union_max = max(
            self.position[0] + self.w - 1, other.position[0] + other.w - 1
        )
        y_union_min = min(self.position[1], other.position[1])
        y_union_max = max(
            self.position[1] + self.h - 1, other.position[1] + other.h - 1
        )
        w_union = x_union_max - x_union_min + 1
        h_union = y_union_max - y_union_min + 1
        position_union = np.array([x_union_min, y_union_min, 1], dtype=np.int0)

        # adjusting the final frame: left strip (self)
        #
        # offset in relation to the new position
        offset = self.position - position_union
        # image
        image_s = np.zeros((h_union, w_union, 3), dtype=np.uint8)
        image_s[
            offset[1] : offset[1] + self.h, offset[0] : offset[0] + self.w
        ] = self.image
        # mask
        mask_s = np.zeros((h_union, w_union), dtype=np.uint8)
        mask_s[
            offset[1] : offset[1] + self.h, offset[0] : offset[0] + self.w
        ] = self.mask
        # border
        left = self.borders["left"] + offset  # SOMAR OU SUBTRAIR?

        # adjusting the final frame: right strip (other)
        #
        # horizontal/vertical offsets in relation to the new position
        offset = other.position - position_union
        # image
        image_o = np.zeros_like(image_s)
        image_o[
            offset[1] : offset[1] + other.h, offset[0] : offset[0] + other.w
        ] = other.image
        # mask
        mask_o = np.zeros_like(mask_s)
        mask_o[
            offset[1] : offset[1] + other.h, offset[0] : offset[0] + other.w
        ] = other.mask
        # border
        right = other.borders["right"] + offset

        # intersection area
        intersection = (mask_s == 255) & (mask_o == 255)

        # stacking
        stacked = cv2.add(image_o, image_s)
        stacked[intersection] = (
            image_s[intersection] // 2 + image_o[intersection] // 2
        )  # average on intersection
        stacked_mask = cv2.add(mask_s, mask_o)

        self.position = position_union
        self.h, self.w = stacked_mask.shape
        self.image = stacked
        self.mask = stacked_mask
        self.borders["left"] = left
        self.borders["right"] = right
        return self

    def crop_center(self, w_crop):
        """Crop the center area of a strip."""

        assert self.artificial_mask

        _, w = self.image.shape[:2]
        x_start = (w - w_crop) // 2

        self.w = w_crop
        self.x += x_start
        self.position[0] = self.x
        self.borders["right"][:, 0] = w_crop
        self.mask = self.mask[:, x_start : x_start + w_crop]
        self.image = self.image[:, x_start : x_start + w_crop]
        return self

    # def vstack(self, other, overlap=False):
    #     ''' Stack vertically with other strip. '''

    #     x_inter_min = 0
    #     x_inter_max = min(self.w, other.w)
    #     x_union_min = 0
    #     x_union_max = max(self.w, other.w)

    #     assert x_inter_min < x_inter_max

    #     # borders coordinates
    #     b1 = self.offsets_b[x_inter_min : x_inter_max]
    #     t2 = other.offsets_t[x_inter_min : x_inter_max]

    #     offset = self.h - np.min(t2 + self.h - b1) + 1
    #     if overlap:
    #         offset -= 1
    #         min_overlap_index = np.abs(t2 + offset - b1).sum()
    #         while np.abs(t2 + offset - b1).sum() < min_overlap_index:
    #             min_overlap_index = np.abs(t2 + offset - b1).sum()
    #             offset -= 1
    #         offset += 1

    #     # new image / mask
    #     # print(self.h, other.h, offset + other.h)
    #     tmp_image = np.zeros((offset + other.h, x_union_max, 3), dtype=np.uint8)
    #     tmp_mask = np.zeros((offset + other.h, x_union_max), dtype=np.uint8)
    #     cv2.add(tmp_image[: self.h, : self.w], self.image, dst=tmp_image[: self.h, : self.w])
    #     cv2.add(tmp_image[offset :, : other.w], other.image, dst=tmp_image[offset :, : other.w])
    #     cv2.add(tmp_mask[: self.h, : self.w], self.mask, dst=tmp_mask[: self.h, : self.w])
    #     cv2.add(tmp_mask[offset :, : other.w], other.mask, dst=tmp_mask[offset :, : other.w])

    #     self.h, self.w = tmp_mask.shape
    #     self.image = tmp_image
    #     self.mask = tmp_mask
    #     self.offsets_l =np.apply_along_axis(first_nonzero, 1, self.mask)
    #     self.offsets_r =np.apply_along_axis(last_nonzero, 1, self.mask)
    #     self.offsets_t = np.apply_along_axis(first_nonzero, 0, self.mask) # top border (hor.) offsets
    #     self.offsets_b = np.apply_along_axis(last_nonzero, 0, self.mask)   # bottom border (hor.) offsets
    #     return self

    # def crop_vertically(self, y1, y2):
    #     ''' Crop the strip vertically from y1 to y2. Indexes follow numpy scheme [y1, y2). '''

    #     self.offsets_l = self.offsets_l[y1 : y2]
    #     self.offsets_r = self.offsets_r[y1 : y2]
    #     x1 = self.offsets_l.min()
    #     x2 = self.offsets_r.max()
    #     self.offsets_l -= x1
    #     self.offsets_r -= x1
    #     self.image = self.image[y1 : y2, x1 : x2 + 1] # height can be different from y2 - y1 for the bottom part of the document
    #     self.mask = self.mask[y1 : y2, x1 : x2 + 1]
    #     self.approx_width = int(np.mean(self.offsets_r - self.offsets_l + 1))
    #     self.w = self.mask.shape[1]
    #     self.h = self.mask.shape[0]

    def vsplit(self, n, shifted=False):
        """Split equally the strip vertically int n pieces (discard the remaining part)."""

        # truncated y-coordinate (remaining pixels are disregarded)
        y_max = self.h - self.h % n

        if not shifted:
            images = np.split(self.image[:y_max], n, axis=0)
            masks = np.split(self.mask[:y_max], n, axis=0)
        else:
            h_crop = y_max // (n + 1)  # crop size if no shift was applied
            offset = h_crop // 2
            y_max = offset + n * h_crop
            images = np.split(self.image[offset:y_max], n, axis=0)
            masks = np.split(self.mask[offset:y_max], n, axis=0)

        strips_list = []
        for image, mask in zip(images, masks):
            strip = Strip(
                image.astype(np.uint8),
                self.index,
                mask.astype(np.uint8),
                self.perc_discard_borders,
            )
            strips_list.append(strip)
        return strips_list


class Strips(object):
    """Strips operations manager."""

    def __init__(self, path=None, filter_blanks=True, blank_tresh=127, strips_list=[]):
        """Strips constructor.

        @path: path to a directory containing strips (in case of load real strips)
        @filter_blanks: true-or-false flag indicating the removal of blank strips
        @uniform_height: keep the strips with the same height
        @height: the new height value (if uniform_height is True)
        @blank_thresh: threshold used in the blank strips filtering
        """

        self.strips = []  # can be altered by load_data
        if path is not None:
            assert os.path.exists(path)
            self.artificial_mask = False  # can be altered by load_data
            self._load_data(path)
        else:
            assert len(strips_list) > 0
            self.strips = [strip.copy() for strip in strips_list]
            self.artificial_mask = self(0).artificial_mask

        # remove low content ('blank') strips
        if filter_blanks:
            self.strips = [
                strip for strip in self.strips if not strip.is_blank(blank_tresh)
            ]
            # after removing the blank spaces
            new_indices = np.argsort([strip.index for strip in self.strips])
            for strip, new_index in zip(self.strips, new_indices):
                strip.index = int(new_index)  # avoid json serialization issues

        self.leftmost_strips = [self(0)]
        self.rightmost_strips = [self(-1)]
        self.sizes = [self.size()]

    def __call__(self, i, index=None):
        """Returns the i-th strip or the strip with the required index."""

        if index is not None:
            # search by index
            for strip in self.strips:
                if strip.index == index:
                    return strip
                    break
            else:
                raise Exception("not found")

        assert i >= -1 and i < len(self.strips)  # i=-1 means the last strip

        # print(i, len(self.strips))
        return self.strips[i]

    def __add__(self, other):
        """Including new strips."""

        num_strips = len(self.strips)
        union = self.copy()
        other = other.copy()
        for strip in other.strips:
            strip.index += num_strips
        union.leftmost_strips += other.leftmost_strips
        union.rightmost_strips += other.rightmost_strips
        union.sizes.append(other.size())
        union.strips += other.strips
        return union

    def copy(self):
        """Copy object."""

        return copy.deepcopy(self)

    def size(self):
        """Number of strips."""

        return len(self.strips)

    def sizes_per_doc(self):
        """Number of strips of each document when mixed strips."""

        return self.sizes

    def shuffle(self):

        random.shuffle(self.strips)
        return self

    def set_permutation(self, order=[], ground_truth_order=False):
        """Set the permutation (order) of the strips."""

        if ground_truth_order:
            self.strips = [
                strip for strip in sorted(self.strips, key=lambda x: x.index)
            ]
        else:
            assert len(order) == self.size()

            self.strips = [self(index=idx) for idx in order]
        return self

    def permutation(self):
        """Return the index permutation of the strips."""

        return [strip.index for strip in self.strips]

    # def extremities(self):
    #     """Return the ground-truth indices of the strips belonging to the documents' extremities."""

    #     left_indices = [strip.index for strip in self.leftmost_strips]

    def pairing(self, i, j):
        """Return a strip composed by the contenation of the strips i and j"""

        return self(i).copy().hstack(self(j))

    def _load_data(self, path, regex_str=".*\d\d\d\d\d\.*"):
        """Load data from disk.

        Strips are images with same basename (and extension) placed in a common
        directory. Example:

        basename="D001" and extension=".jpg" => strips D00101.jpg, ..., D00130.jpg.
        """

        path_images = "{}/strips".format(path)
        path_masks = "{}/masks".format(path)
        regex = re.compile(regex_str)

        # loading images
        fnames = sorted(
            [fname for fname in os.listdir(path_images) if regex.match(fname)]
        )
        images = []
        for fname in fnames:
            image = cv2.cvtColor(
                cv2.imread("{}/{}".format(path_images, fname)), cv2.COLOR_BGR2RGB
            )
            images.append(image)

        # load masks
        masks = []
        if os.path.exists(path_masks):
            for fname in fnames:
                mask = np.load(
                    "{}/{}.npy".format(path_masks, os.path.splitext(fname)[0])
                )
                masks.append(mask)
        else:
            masks = len(images) * [None]
            self.artificial_mask = True

        for index, (image, mask) in enumerate(zip(images, masks), 1):
            strip = Strip(image, index, mask)
            self.strips.append(strip)

    # def composition(
    #     self, displacements=None, pre_alignment='trivial', method='simple',
    #     threshold_icp=10, iterations_icp=30, sample_factor_icp=1, highlight=False,
    #     max_theta_ex=5.0, num_angles_ex=20,
    #     highlight_idx=0, alpha=0.5, color=(255, 0, 255)
    # ):

    def composition(
        self,
        displacements=None,
        pre_alignment=None,
        fine_alignment=None,
        correct_rotation_first=False,
        correct_rotation_all=False,
        threshold=10,
        iterations=10,
        sample_factor=1,  # icp parameters
        max_theta=5.0,
        num_angles_per_degree=1,  # best rot parameters
        max_dx=3,
        max_dy=3,  # best trans parameters
        highlight=False,
        highlight_idx=0,
        alpha=0.5,
        color=(255, 0, 255),
    ):
        """Return the entire collection as a single strip object."""

        assert pre_alignment in [None, "trivial", "centroid"]
        assert fine_alignment in [None, "best-rot", "best-trans", "icp"]

        strips_list = [strip.copy() for strip in self.strips]
        num_strips = self.size()

        if displacements is not None:
            assert displacements.shape == (num_strips,)

            # ignores the first strip
            for i in range(1, len(strips_list)):
                strips_list[i].vert_shift(displacements[i])

        if highlight:
            current = strips_list[highlight_idx]
            overlay = np.zeros_like(current.image)
            mask = current.mask > 0
            overlay[mask] = color
            current.image = cv2.addWeighted(overlay, alpha, current.image, 1 - alpha, 0)

        if correct_rotation_all:
            for strip in strips_list:
                strip.correct_rotation(max_theta, num_angles_per_degree, sample_factor)
        else:
            if correct_rotation_first:
                strips_list[0].correct_rotation(
                    max_theta, num_angles_per_degree, sample_factor
                )

        stacked = strips_list[0]
        for strip in strips_list[1:]:
            stacked.hstack(
                strip,
                pre_alignment=pre_alignment,
                fine_alignment=fine_alignment,
                threshold=threshold,
                iterations=iterations,
                sample_factor=sample_factor,
                max_theta=max_theta,
                num_angles_per_degree=num_angles_per_degree,
                max_dx=max_dx,
                max_dy=max_dy,
            )
        return stacked

    def vsplit(self, n=1, shifted=False):  # not used at the moment

        strips_splits = [strip.vsplit(n, shifted) for strip in self.strips]
        blocks = []
        for row in range(n):
            strips_list = [strip_split[row] for strip_split in strips_splits]
            blocks.append(Strips(strips_list=strips_list, filter_blanks=False))
        return blocks

    # def post_processed_image_composition(self, order=None, ground_truth_order=False, displacements=None, overlap=False, offset=0, num_crops=50):

    #     composed = self.image_composition(order=order, displacements=displacements, overlap=overlap)
    #     if num_crops == 1:
    #         return composed

    #     # extracting crops
    #     h = composed.h
    #     cuts = np.array(num_crops * [h // num_crops])
    #     cuts[: h % num_crops] += 1
    #     cuts = cuts.cumsum() + offset
    #     crops = []
    #     y1 = 0
    #     for y2 in cuts:
    #         crop = self.copy().crop_vertically(y1, y2).image_composition(
    #             order=order, ground_truth_order=ground_truth_order,
    #             displacements=displacements, overlap=overlap
    #         )
    #         crops.append(crop)
    #         y1 = y2

    #     # result image
    #     stacked = crops[0].copy()
    #     for crop in crops[1 :]:
    #         stacked.vstack(crop, overlap=overlap)
    #     return stacked

    # def crop_vertically(self, y1, y2):
    #     ''' Crop the strips vertically from y1 to y2. Indexes follow numpy scheme [y1, y2). '''

    #     for strip in self.strips:
    #         strip.crop_vertically(y1, y2)
    #     return self
