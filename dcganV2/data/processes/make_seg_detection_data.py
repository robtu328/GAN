import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper

from concern.config import State
from .data_process import DataProcess


class MakeSegDetectionData(DataProcess):
    r'''
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''
    min_text_size = State(default=8)
    shrink_ratio = State(default=0.4)

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def process(self, data):
        '''
        requied keys:
            image, polygons, ignore_tags, filename
        adding keys:
            mask
        '''
        image = data['image']
        polygons = data['polygons']
        polygons_char = data['polygons_char']
        ignore_tags = data['ignore_tags']
        ignore_tags_char = data['ignore_tags_char']
        image = data['image']
        filename = data['filename']

        h, w = image.shape[:2]
        if data['is_training']:
            polygons, ignore_tags = self.validate_polygons(
                polygons, ignore_tags, h, w)

            polygons_char, ignore_tags_char = self.validate_polygons(
                polygons_char, ignore_tags_char, h, w)
            
            
        gt = np.zeros((1, h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        gt_char = np.zeros((1, h, w), dtype=np.float32)
        mask_char = np.ones((h, w), dtype=np.float32)
       
        
        for i in range(len(polygons)):
            polygon = polygons[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            # height = min(np.linalg.norm(polygon[0] - polygon[3]),
            #              np.linalg.norm(polygon[1] - polygon[2]))
            # width = min(np.linalg.norm(polygon[0] - polygon[1]),
            #             np.linalg.norm(polygon[2] - polygon[3]))
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(
                    np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                polygon_shape = Polygon(polygon)
                distance = polygon_shape.area * \
                    (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
                subject = [tuple(l) for l in polygons[i]]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                if shrinked == []:
                    cv2.fillPoly(mask, polygon.astype(
                        np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                cv2.fillPoly(gt[0], [shrinked.astype(np.int32)], 1)


        for i in range(len(polygons_char)):
            polygon_char = polygons_char[i]
            height = max(polygon_char[:, 1]) - min(polygon_char[:, 1])
            width = max(polygon_char[:, 0]) - min(polygon_char[:, 0])

            if ignore_tags_char[i]: #or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask_char, polygon_char.astype(
                    np.int32)[np.newaxis, :, :], 0)
                ignore_tags_char[i] = True
            else:
                polygon_shape = Polygon(polygon_char)
                distance = polygon_shape.area * \
                    (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
                subject = [tuple(l) for l in polygons_char[i]]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                if shrinked == []:
                    cv2.fillPoly(mask_char, polygon_char.astype(
                        np.int32)[np.newaxis, :, :], 0)
                    ignore_tags_char[i] = True
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                cv2.fillPoly(gt_char[0], [shrinked.astype(np.int32)], 1)

        if filename is None:
            filename = ''
        data.update(image=image,
                    polygons=polygons,
                    gt=gt, mask=mask, 
                    polygons_char=polygons_char,
                    gt_char=gt_char, mask_char=mask_char,
                    filename=filename)
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (polygon[next_index, 1] + polygon[i, 1])

        return edge / 2.

