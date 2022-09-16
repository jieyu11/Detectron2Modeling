import cv2
import logging

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
					level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoLoader:
	def __init__(self, videofile, nframes=1e6):
		"""Read a video file .

		Args:
			videofile ([type]): full path to the input video.
			nframes ([type], optional): [description]. Defaults to 1e6.

		Returns:
			[list]: the list of frames read from the input video.
		"""
		cap = cv2.VideoCapture(videofile)
		self._frames = []
		idx = 0
		while(cap.isOpened()):
			if idx >= nframes:
				break

			ret, frame = cap.read()
			if not ret:
				logger.error("Can't receive frame (stream end?). Exiting ...")
				break
			# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			# frame = frame[:, :, ::-1]
			self._frames.append(frame)
			idx += 1
		cap.release()
		cv2.destroyAllWindows()
		logger.info("Read %d frames from %s." % (len(self._frames), videofile))
	
	@property
	def frames(self):
		return self._frames
