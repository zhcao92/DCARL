import rospy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

def check_color(value):
    if value > 255.:
        return 1
    elif value < 0:
        return 0.
    else:
        return value / 255.

def check_alpha(value):
    if value > 1:
        return 1
    elif value < 0.1:
        return 0.1
    else:
        return value

# Colors should be in order r, g, b, a
def parse_color(in_color):
    color = ColorRGBA()
    if len(in_color) == 4:
        color.r = check_color(in_color[0])
        color.g = check_color(in_color[1])
        color.b = check_color(in_color[2])
        color.a = check_alpha(in_color[3])
    else:
        rospy.logerr("Cannot resolve color value. Check configuration please!")
    return color

class RvizVisualizer():

    @staticmethod
    def draw_two_points_line(point1, point2, width=1, duration=0.5):
        tempmarkerarray = MarkerArray()
        tempmarker = Marker()
        tempmarker.header.frame_id = "map"
        tempmarker.header.stamp = rospy.Time.now()
        tempmarker.ns = "zzz/decision"
        tempmarker.id = 1
        tempmarker.type = Marker.LINE_STRIP
        tempmarker.action = Marker.ADD
        tempmarker.scale.x = width
        start_point = Point()
        start_point.x = point1[0]
        start_point.y = point1[1]
        start_point.z = 2
        end_point = Point()
        end_point.x = point2[0]
        end_point.y = point2[1]
        end_point.z = 2
        tempmarker.points.append(start_point)
        tempmarker.points.append(end_point)
        tempmarker.color.r = 0.0
        tempmarker.color.g = 1.0
        tempmarker.color.b = 1.0
        tempmarker.color.a = 1
        tempmarker.lifetime = rospy.Duration(duration)
        tempmarkerarray.markers.append(tempmarker)
        return tempmarkerarray

    @staticmethod
    def draw_line_from_points(points_list, color=(0,1,1), width=3):
        tempmarkerarray = MarkerArray()
        tempmarker = Marker()
        tempmarker.header.frame_id = "map"
        tempmarker.header.stamp = rospy.Time.now()
        tempmarker.ns = "zzz/decision"
        tempmarker.id = 1
        tempmarker.type = Marker.LINE_STRIP
        tempmarker.action = Marker.ADD
        tempmarker.scale.x = width
        for point in points_list:
            point_marker = Point()
            point_marker.x = point[0]
            point_marker.y = point[1]
            point_marker.z = 0
            tempmarker.points.append(point_marker)
        tempmarker.color.r = color[0]
        tempmarker.color.g = color[1]
        tempmarker.color.b = color[2]
        tempmarker.color.a = 0.5
        tempmarker.lifetime = rospy.Duration(0.5)
        tempmarkerarray.markers.append(tempmarker)
        return tempmarkerarray

    @staticmethod
    def draw_circle(center_x, center_y, radius, duration=0.5):
        tempmarkerarray = MarkerArray()
        tempmarker = Marker()
        tempmarker.header.frame_id = "map"
        tempmarker.header.stamp = rospy.Time.now()
        tempmarker.ns = "zzz/decision"
        tempmarker.id = 1
        tempmarker.type = Marker.CYLINDER
        tempmarker.action = Marker.ADD
        tempmarker.scale.x = radius
        tempmarker.scale.y = radius
        tempmarker.scale.z = radius
        tempmarker.color.r = 0.0
        tempmarker.color.g = 1.0
        tempmarker.color.b = 0.5
        tempmarker.color.a = 0.5
        tempmarker.lifetime = rospy.Duration(duration)
        tempmarker.pose.position.x = center_x
        tempmarker.pose.position.y = center_y
        tempmarkerarray.markers.append(tempmarker)
        return tempmarkerarray

    @staticmethod
    def draw_words(text, pos_x, pos_y, scale=0.5, duration=0.5):
        tempmarkerarray = MarkerArray()
        tempmarker = Marker()
        tempmarker.header.frame_id = "map"
        tempmarker.header.stamp = rospy.Time.now()
        tempmarker.ns = "zzz/decision"
        tempmarker.id = 1
        tempmarker.type = Marker.TEXT_VIEW_FACING
        tempmarker.action = Marker.ADD
        tempmarker.scale.z = scale
        tempmarker.color.r = 0.0
        tempmarker.color.g = 1.0
        tempmarker.color.b = 0.5
        tempmarker.color.a = 1
        tempmarker.lifetime = rospy.Duration(duration)
        tempmarker.pose.position.x = pos_x
        tempmarker.pose.position.y = pos_y
        tempmarker.pose.position.z = 2
        tempmarker.text = text
        tempmarkerarray.markers.append(tempmarker)
        return tempmarkerarray
