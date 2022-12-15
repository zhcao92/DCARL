
import rospy
import logging

class LogBridge(logging.Handler):
    MAP = {
        logging.DEBUG       :rospy.logdebug,
        logging.INFO        :rospy.loginfo,
        logging.WARNING     :rospy.logwarn,
        logging.ERROR       :rospy.logerr,
        logging.CRITICAL    :rospy.logfatal
    }

    def emit(self, record):
        try:
            self.MAP[record.levelno]("%s: %s" % (record.name, record.getMessage()))
        except KeyError:
            rospy.logerr("unknown log level %s LOG: %s: %s" % (record.levelno, record.name, record.msg))
