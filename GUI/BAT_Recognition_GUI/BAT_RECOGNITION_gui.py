import tkinter as tk
from tkinter import ttk
from tkinter import *
import datetime
from tkinter.filedialog import askdirectory, askopenfilename
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam
from PIL import Image,ImageTk
import threading
import numpy as np

class goods_counting_gui:
    def __init__(self):
        super().__init__()
        self.create_gui()
    def create_gui(self):
        self.window = tk.Tk()
        self.window.title('COMP90055 Bats Classification & Detection - by Yifu Tang')
        self.label = tk.Label(self.window, text='Current Time：', bg='green', font=30)  # text是要显示的内容
        self.label.grid(row=0, column=0)
        self.cur_time = tk.Label(self.window, text='%s%d'%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:'),
                                                          datetime.datetime.now().microsecond//100000), font=30)
        self.cur_time.grid(row=0, column=1)
        self.window.after(100, self.update_time)

        # self.project = tk.Text(self.window, width =100,height=20)
        # self.project.insert(END, "COMP90055 Dissertation Project", 'big')
        # self.project.grid(row=0, column=0, columnspan=2, padx=50, pady=20)

        self.img_path = StringVar()
        self.confidence = StringVar()
        self.label_yolo_obj = tk.Label(self.window, text='Image Path: ')
        self.label_yolo_obj.grid(row=1, column=0, pady=6)
        self.entry2 = tk.Entry(self.window, textvariable=self.img_path, width=50)
        self.entry2.grid(row=1, column=1, padx=5, pady=6)

        # self.button_yolo_obj = tk.Button(self.window, text='Choose Image', bg='pink', relief=tk.RAISED, width=14,
                                        # height=1, command=self.get_img_path)
        self.button_yolo_obj = ttk.Button(self.window, text='Choose Image', command=self.get_img_path)
        self.button_yolo_obj.grid(row=1, column=2, padx=5, pady=20)

        self.button_start = ttk.Button(self.window, text='Start Classification', command=self.cls_start)
        self.button_start.grid(row=2, column=2, padx=10, pady=6)

        # self.button_start = tk.Button(self.window, text='Start Detection', bg='gold', relief=tk.RAISED,
        #                               width=13, height=1, command=self.start)
        self.button_start = ttk.Button(self.window, text='Start Detection', command=self.obj_start)
        self.button_start.grid(row=3, column=2, padx=10, pady=6)

        self.l1 = tk.Label(self.window, text='Confidence Threshold', bg='yellow', font=60)
        self.l1.grid(row=3, column=3, padx=2, pady=6, sticky=tk.NW)
        self.Con = tk.Entry(self.window, textvariable=self.confidence, width=5)
        self.Con.grid(row=3, column=4, padx=2, pady=6)
        

        # self.button_start = tk.Button(self.window, text='Close', bg='red', relief=tk.RAISED, width=13, height=1,
        #                               command=self.stop)
        self.button_start = ttk.Button(self.window, text='Close', command=self.stop)
        self.button_start.grid(row=6, column=4, padx=10, pady=10)

        self.frm_ = tk.Frame(bg='white')
        self.frm_.grid(row=5, column=1, padx=5)

        self.result_show = tk.Label(self.frm_, bg='white')
        self.result_show.grid(row=5, column=1, padx=5)

        # self.output = tk.Label(self.window, text='Detecting results Output', bg='yellow', font=60)
        # self.output.grid(row=3, column=2, padx=5, pady=5, sticky=tk.NW)

        # self.text = tk.Text(self.window,width =20,height=10)
        # self.text.grid(row=3, column=3, pady=5, sticky=tk.NW)

    def get_img_path(self):
        # self.path1 = askdirectory()
        self.path1 = askopenfilename()
        self.img_path.set(self.path1)

        frame = cv2.imread(self.path1) # ndarray
        B, G, R = cv2.split(frame)
        frame = cv2.merge([R, G, B])
        img = Image.fromarray(frame)  # 类型是PIL.Image.Image
        img = self.resize(img)
        imgtk = ImageTk.PhotoImage(img)
        self.result_show.config(image=imgtk)
        self.result_show.img = imgtk


    def update_time(self):
        self.cur_time.config(text=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.window.after(500, self.update_time)
    
    def cls_start(self):
        # self.stop_flag = False
        t = threading.Thread(target=self.run_classification())
        t.setDaemon(True)
        t.start()

    def obj_start(self):
        # self.stop_flag = False
        t = threading.Thread(target=self.run_detection())
        t.setDaemon(True)
        t.start()

    def stop(self):
        # self.stop_flag = True
        # self.window.quit() 退出窗体
        # self.result_show.quit()
        # frame = cv2.imread(self.path1)
        # B, G, R = cv2.split(frame)
        # frame = cv2.merge([R, G, B])
        # img = Image.fromarray(frame)  # 类型是PIL.Image.Image
        # img = self.resize(img)
        # imgtk = ImageTk.PhotoImage(img)
        # self.result_show.config(image=imgtk)
        # self.result_show.img = imgtk
        # for widget in self.frm_.winfo_children():
        #     widget.destroy()
        # self.result_show.grid_forget()
        exit()
        #pass



    def resize(self, image):
        im = image
        self.new_size = (800, 800)
        im.thumbnail(self.new_size,Image.ANTIALIAS)  # thumbnail() 函数是制作当前图片的缩略图, 参数size指定了图片的最大的宽度和高度。
        return im

    def LoadImage(self, image):
        plot = cv2.imread(image)
        image = cv2.resize(plot, (224, 224))
        image = image[:, :, [2, 1, 0]]
        image = image.astype('float64')
        image /= 255.0  # normalize to [0,1] range

        return plot, np.asarray(image)

    def run_classification(self):
        model_path = "model_num2.h5"
        model = load_model(model_path)
        model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])
        label_dict = {0: "Egyptian Fruit Bat", 1: "Giant Golden-Crowned Flying-Fox Bat", 2: "Indiana Bat",
                3: "Kittis Hog-Nosed Bat", 4: "Little Brown Bat", 5: "Vampire Bat"}
        
        plot, image = self.LoadImage(self.path1)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        res = np.argmax(model.predict(image))
        text = label_dict[int(res)]


        font_scale = 1.5
        font = cv2.FONT_HERSHEY_PLAIN

        # set the rectangle background to white
        rectangle_bgr = (255, 255, 255)
        # get the width and height of the text box
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
        # set the text start position
        text_offset_x = 20
        text_offset_y = plot.shape[0] - 20
        # make the coords of the box with a small padding of two pixels
        box_coords = ((text_offset_x-5, text_offset_y+5), (text_offset_x + text_width + 5, text_offset_y - text_height - 5))
        cv2.rectangle(plot, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        cv2.putText(plot, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)

        # plot = cv2.putText(plot,pic_class,(0,40),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)

        frame = plot # ndarray
        B, G, R = cv2.split(frame)
        frame = cv2.merge([R, G, B])
        img = Image.fromarray(frame)  # 类型是PIL.Image.Image
        img = self.resize(img)
        imgtk = ImageTk.PhotoImage(img)
        self.result_show.config(image=imgtk)
        self.result_show.img = imgtk


    def run_detection(self):
        weightsPath = "yolov3_bat_final.weights"
        configPath = "yolov3_bat.cfg"
        labelsPath = "bat.names"
        LABELS = open(labelsPath).read().strip().split("\n")  # 物体类别
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")  # 颜色
        boxes = []
        confidences = []
        classIDs = []
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        image = cv2.imread(self.path1)
        # print(self.path1)
        (H, W) = image.shape[:2]

        # 得到 YOLO需要的输出层
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # 从输入图像构造一个blob，然后通过加载的模型，给我们提供边界框和相关概率
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # 在每层输出上循环
        for output in layerOutputs:
            # 对每个检测进行循环
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # 过滤掉那些置信度较小的检测结果
                con = self.confidence.get()
                if confidence >= float(con):
                    # 框后接框的宽度和高度
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # 边框的左上角
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # 更新检测出来的框
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # 极大值抑制
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.3)
        # print('idxs =', idxs)
        if len(idxs) > 0:
            for i in idxs.flatten():  # 把一列拉成一行,比如[[1] \ [0] \[2] ]  变成[1 0 2 ]
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # 在原图上绘制边框和类别
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x+20, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # cv2.imshow("Image", image)
        # cv2.imwrite('result.jpg', image)
        frame = image # ndarray
        B, G, R = cv2.split(frame)
        frame = cv2.merge([R, G, B])
        img = Image.fromarray(frame)  # 类型是PIL.Image.Image
        img = self.resize(img)
        imgtk = ImageTk.PhotoImage(img)
        self.result_show.config(image=imgtk)
        self.result_show.img = imgtk
        # if len(idxs) > 0:
        #     for i in idxs.flatten():
        #         if LABELS[classIDs[i]] == 'insulator':
        #             self.text.insert(INSERT,'Detecting result is: ', LABELS[classIDs[i]])

        cv2.imwrite('dec_result.jpg', image)
        # cv2.waitKey(5)

    def text_insert(self):
        pass

    # @staticmethod
    # def thread_it(func, *args):
    #     t = threading.Thread(target=func, args=args)
    #     t.setDaemon(True)
    #     t.start()

if __name__ == '__main__':
    goods_counting_gui()
    mainloop()
