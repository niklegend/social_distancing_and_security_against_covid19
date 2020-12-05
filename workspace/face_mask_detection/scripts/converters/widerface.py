import os

import scipy.io

from .tokitticonverter import ToKittiConverter, Category


class WiderFaceToKittiConverter(ToKittiConverter):

    def __init__(
            self,
            kitti_base_dir,
            kitti_image_size,
            images_dir,
            annotations_path,
            limit,
            stage,
            verbose
    ):
        super(WiderFaceToKittiConverter, self).__init__(kitti_base_dir, limit, kitti_image_size, verbose, stage)

        self.annotations_path = annotations_path
        self.data = scipy.io.loadmat(self.annotations_path)
        self.file_names = self.data.get('file_list')  # File Name
        self.event_list = self.data.get('event_list')  # Folder Name
        self.bbox_list = self.data.get('face_bbx_list')  # Bounding Boxes
        self.label_list = self.data.get('occlusion_label_list')
        self.images_dir = images_dir
        self.len_dataset = len(self.file_names)

    def __call__(self):
        # pick_list = ['19--Couple', '13--Interview', '16--Award_Ceremony','2--Demonstration', '22--Picnic']
        # Use following pick list for more image data
        pick_list = ['2--Demonstration', '4--Dancing', '5--Car_Accident', '15--Stock_Market', '23--Shoppers',
                     '27--Spa', '32--Worker_Laborer', '33--Running', '37--Soccer',
                     '47--Matador_Bullfighter', '57--Angler', '51--Dresses', '46--Jockey',
                     '9--Press_Conference', '16--Award_Ceremony', '17--Ceremony',
                     '20--Family_Group', '22--Picnic', '25--Soldier_Patrol', '31--Waiter_Waitress',
                     '49--Greeting', '38--Tennis', '43--Row_Boat', '29--Students_Schoolkids']

        for event_idx, event in enumerate(self.event_list):
            directory = event[0][0]
            if any(ele in directory for ele in pick_list):
                for im_idx, im in enumerate(self.file_names[event_idx][0]):
                    im_name = im[0][0]
                    image_name = os.path.join(directory, im_name + '.jpg')
                    face_bbx = self.bbox_list[event_idx][0][im_idx][0]
                    category_id = self.label_list[event_idx][0][im_idx][0]
                    #  print face_bbx.shape
                    bboxes = []
                    class_names = []
                    if self.count_no_mask < self.no_mask_limit:
                        for i in range(face_bbx.shape[0]):
                            xmin = int(face_bbx[i][0])
                            ymin = int(face_bbx[i][1])
                            xmax = int(face_bbx[i][2]) + xmin
                            ymax = int(face_bbx[i][3]) + ymin
                            # Consider only Occlusion Free masks
                            if category_id[i][0] == 0:
                                category_name = Category.NO_MASK
                                bboxes.append((xmin, ymin, xmax, ymax))
                                class_names.append(category_name)

                        if bboxes and len(bboxes) < 4:
                            # print("Len of BBox:{} in Image:{}".format(len(bboxes),im_name))
                            self.write_example(
                                image_path=os.path.join(self.images_dir, image_name),
                                class_names=class_names,
                                bboxes=bboxes)

        return self.count_mask, self.count_no_mask
