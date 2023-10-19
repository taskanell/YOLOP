#!/usr/bin/env python

from threading import Thread, Event
from carla import VehicleControl
import argparse
import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
import carla

#import ros_compatibility as roscomp
import rospy
import cv2
import message_filters
#from ros_compatibility.node import CompatibleNode
#from ros_compatibility.qos import QoSProfile, DurabilityPolicy

from carla_msgs.msg import CarlaStatus
from carla_msgs.msg import CarlaEgoVehicleInfo
from carla_msgs.msg import CarlaEgoVehicleStatus
from carla_msgs.msg import CarlaEgoVehicleControl
from carla_msgs.msg import CarlaLaneInvasionEvent
from carla_msgs.msg import CarlaCollisionEvent
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool,Empty

from ultralytics import YOLO
import torch
import numpy as np
import time
import sys
import math
#sys.append('/home/iccs/git/panagiotis/yolov5')
#from models.common import AutoShape


class RosControl(object):

    def __init__(self):

        #super(RosControl, self).__init__("RosControl")

        self.role_name = rospy.get_param("role_name", "hero")
        #self.shutdown = Event()
        #self.autopilot_enabled = Event()
        self.autopilot_enabled = False
        self.distance_lower = False
        self.control_commands_thread = Thread(target=self.control_commands)

        self.cv_bridge = CvBridge()
        #self.counter_d = 0
        #self.counter_r = 0
        self.counter = 0
        #self.rate = rospy.Rate(0.001)

        self.depth_image = None
        self.rgb_image = None
        self.ego_velocity = None

        self.rgb_stamp = None
        self.depth_stamp = None
        self.ego_velocity_stamp = None

        self.first_throttle = True

        self.autopilot_switch_by_the_driver_subscriber = rospy.Subscriber(
            "/carla/{}/ego_switch_to_autopilot".format(self.role_name),
            Empty,
            self.trigger_ros_control)
        '''
        self.front_rgb_image_subscriber = rospy.Subscriber(
            "/carla/{}/front/RgbCamera/image".format(self.role_name),
            Image,
            self.get_rgb_image)
        
        self.front_depth_image_subscriber = rospy.Subscriber(
            "/carla/{}/front/DepthCamera/image".format(self.role_name),
            Image,
            self.get_depth_image)
        '''
        self.front_rgb_image_subscriber = message_filters.Subscriber("/carla/{}/front/RgbCamera/image".format(self.role_name),Image)

        self.front_depth_image_subscriber = message_filters.Subscriber("/carla/{}/front/DepthCamera/image".format(self.role_name),Image)

        self.vehicle_status = message_filters.Subscriber("/carla/{}/vehicle_status_ev".format(self.role_name),CarlaEgoVehicleStatus)

        self.autopilot_state_subscriber = rospy.Subscriber(
            "/carla/{}/autopilot_status".format(self.role_name),
            Bool,
            self.get_autopilot_status)
        
        self.vehicle_control_publisher =rospy.Publisher(
            "/carla/{}/vehicle_control_cmd_ev".format(self.role_name),
            CarlaEgoVehicleControl,
            queue_size=1)
        
        self.ts = message_filters.TimeSynchronizer([self.front_rgb_image_subscriber, self.front_depth_image_subscriber,self.vehicle_status], 10)
        #self.ts = message_filters.ApproximateTimeSynchronizer([self.front_rgb_image_subscriber, self.front_depth_image_subscriber], slop=2, queue_size=1)
        self.ts.registerCallback(self.get_rgb_depth_camera)
        
    
    def trigger_ros_control(self,data):
        
        if self.autopilot_enabled is True:
            self.autopilot_enabled = False
        else:
            self.autopilot_enabled = True
            self.first_throttle = True 
   
    def get_rgb_depth_camera(self,rgb_img_msg,depth_img_msg,status_msg):
        
        self.counter+= 1
        #self.new_frame = True
        try:
            self.depth_image = self.cv_bridge.imgmsg_to_cv2(depth_img_msg,"passthrough")
        except CvBridgeError as e:
            rospy.logerr("Error: {}".format(e))
        
        try:
            self.rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_img_msg,"rgb8")
        except CvBridgeError as e:
            rospy.logerr("Error: {}".format(e))
        
        self.ego_velocity = 3.6 * status_msg.velocity

        self.ego_velocity_stamp = status_msg.header.stamp
        self.rgb_stamp = rgb_img_msg.header.stamp
        self.depth_stamp = depth_img_msg.header.stamp

    def get_autopilot_status(self, status):
        """
        get the autopilot status
        """
        
        self.autopilot_enabled = status.data
        rospy.loginfo("Manual Control started with autopilot set to {}".format(status.data))
        self.autopilot_state_subscriber.unregister() 
        ### this will cause broken pipe error warning because the publisher is maybe still sending message when unregister() happens
    
    def control_commands(self):

        rospy.loginfo("Steer")
        self.vehicle_control_publisher.publish(CarlaEgoVehicleControl(steer=-0.5,brake=0.0,throttle=0.0))
        rospy.sleep(0.9)
        rospy.loginfo("Started breaking 1")
        self.vehicle_control_publisher.publish(CarlaEgoVehicleControl(steer=0.0,brake=0.2,throttle=0.0))
        rospy.sleep(0.5)
        rospy.loginfo("Steer right")
        self.vehicle_control_publisher.publish(CarlaEgoVehicleControl(steer=0.5,brake=0.0,throttle=0.0))
        rospy.sleep(0.9)
        rospy.loginfo("Started breaking 2")
        self.vehicle_control_publisher.publish(CarlaEgoVehicleControl(steer=0.0,brake=1.0,throttle=0.0))
        rospy.sleep(0.5)
        self.distance_lower = False
    
    def control_vehicle(self):
        
        if self.rgb_image is not None:
            #rospy.loginfo ("image taken")
            result = model(self.rgb_image)
            self.rgb_image = None
            img = np.squeeze(result.render())
            width = img.shape[1]
            height = img.shape[0]
            result = np.array(result.xyxy[0].cpu())
            #for r in result:
                #print (r)
            widths = result [:,2] - result [:,0] 
            result = result[:,2:4]
            result [:,0] =  result [:,0] - widths/2
            result = result.astype(np.int64)
            #print(result)
            if np.size(result) != 0:
               result[0][0] = max(0, min(result[0][0], height - 1))
               result[0][1] = max(0, min(result[0][1], width - 1))
            #print(result)
            #result = result -1    ##HAVE TO CHECK THIS
            img_result = np.flip (result,axis=1)
            img_result = tuple(map(tuple, img_result))
            result = tuple(map(tuple, result))

            for r in result:
                #print (r) 
                cv2.line(img, (width//2,height), r ,color = (255, 0, 0), thickness=2)
            cv2.imshow('YOLO', img)
            cv2.waitKey(1)

            for i,res in enumerate(img_result):
                #rospy.loginfo(self.rgb_stamp == self.depth_stamp)
                #rospy.loginfo(self.rgb_stamp == self.ego_velocity_stamp)
                #print('Num:{}'.format(len(img_result)))
                rospy.loginfo('Frame{}: Object{}: Distance:{}'.format(self.counter,i+1,self.depth_image[res]))
                #print('Velocity: {}'.format(self.ego_velocity))
                #print('X Pixel Distance:{}'.format(abs(res[1]-width//2)))
                if self.autopilot_enabled is True:

                    #print("Here", self.distance_lower)

                    if self.first_throttle:
                        self.vehicle_control_publisher.publish(CarlaEgoVehicleControl(throttle=0.8))
                        rospy.loginfo("First Throttle")
                        #time.sleep(0.5)
                        self.first_throttle = False
                        #self.distance_lower = False
                    
                    #if self.depth_image[res] < 25:
                        #count = self.counter
                        #t1= time.time()
            
                    if self.depth_image[res] < 11 and self.distance_lower is False: #and abs(res[1]-width//2) < 100
                        self.distance_lower = True
                        self.control_commands_thread = Thread(target=self.control_commands)
                        self.control_commands_thread.start()

        else:
            return
                
            '''
            if self.depth_image[res] < 5: #and abs(res[1]-width//2) < 100
            '''
            
    def control_vehicle_yolop(self):
        if self.rgb_image is not None:
            rospy.loginfo ("image taken")
            torch.hub.load(src,'yolop', source ='local', mod = model, image=self.rgb_image, device='0')
            self.rgb_image = None
        else:
            return
		
    def destroy(self):

        rospy.loginfo("Shutting down...")
        #self.shutdown.set()
        self.vehicle_control_publisher.unregister()
        self.autopilot_switch_by_the_driver_subscriber.unregister()
        #self.autopilot_state_subscriber.unregister()
        cv2.destroyAllWindows()
        # del previous image entries
        # os.system("rm -f /media/cinnamon/Files/ros_images/*")
        #os.system("rm -rf /run/ros_imgs")

def main():

    rospy.init_node("ros_control")
    node = rospy.get_name()
    rospy.loginfo("{} node started".format(node))

    global r , model , src

    #source = '/home/iccs/git/driving-simulator/runs/detect/yolov8m_custom_0.22/weights/best.pt'   #'/media/cinnamon/Files/best.pt'
    #src = '/media/cinnamon/Files/yolov5'
    #src = '/home/ghatz/git/yolov5'
    #src = '/home/iccs/git/yolov5'
    src = '/home/iccs/git/' + opt.model
    #model = YOLO('yolov8n')
    #model = YOLO("yolov8n.pt")

    #model = torch.hub.load(src,'custom','yolov5s.pt', source ='local')
    #model = torch.hub.load(src,'custom', opt.model + str('s.pt') , source ='local', device='0')
    #yolop = True
    #model.imgsz = (480,480)
    #model.conf = 0.5
    #model.iou = 0.1
    #model.classes = [1,2,3,5,7]
    #model.cuda()
    #rospy.loginfo(model.device)
    #model.device = torch.device(0)
    #cv_bridge = CvBridge()

    try:

        ros_control_node = RosControl()

        rospy.on_shutdown(ros_control_node.destroy)

        if opt.model=='yolov5':
            model = torch.hub.load(src,'custom', opt.model + str('s.pt') , source ='local', device=opt.device)
            model.conf = 0.5
            model.cuda()
            while not rospy.is_shutdown():
               ros_control_node.control_vehicle()
               rospy.sleep(0.01)
        else:
            model = torch.hub.load(src, opt.model, source ='local', device=opt.device)
            while not rospy.is_shutdown():
               ros_control_node.control_vehicle_yolop()
               rospy.sleep(0.01)
    
    except rospy.ROSInterruptException:
        rospy.loginfo("User requested shut down.")

    finally:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov5', help='yolo model name')
    parser.add_argument('--device', type=str, default='0', help='device name')
    opt = parser.parse_args()
    main()
