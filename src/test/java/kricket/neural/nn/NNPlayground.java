package kricket.neural.nn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import kricket.neural.mnist.Image;
import kricket.neural.mnist.Loader;
import kricket.neural.util.NNOptions;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class NNPlayground {
	final static int IMGSIZE = Image.HEIGHT*Image.WIDTH;
	List<Image> trainingImages, testImages;
	long startTime;
	
	@Before
	public void loadImages() throws IOException {
		trainingImages = Loader.loadImages(Loader.IMAGES_TRAINING_FILE, Loader.LABELS_TRAINING_FILE);
		testImages = Loader.loadImages(Loader.IMAGES_10K_FILE, Loader.LABELS_10K_FILE);
		startTime = System.currentTimeMillis();
	}
	
	@After
	public void logTime() {
		System.out.println(String.format("Total time: %.3fs", (System.currentTimeMillis() - startTime)*0.001));
	}
	
	private void totals(NN nn) {
		nn.getOptions().log.info("\n=====================================\nTotals:");
		nn.getOptions().log.info("Training images:");
		nn.calc_error(trainingImages);
		nn.getOptions().log.info("Test images:");
		nn.calc_error(testImages);
	}
	
	private NNOptions getOpts() {
		NNOptions opts = new NNOptions();
		opts.calcErrorsAfterEpochs = true;
		opts.logDropout = true;
		opts.logEpochs = true;
		opts.logIncorrectAnswers = false;
		opts.summarizeSGD = false;
		return opts;
	}

	@Test
	public void standard() {
		NN nn = new NN(getOpts(), IMGSIZE, 30, 10);
		nn.SGD(trainingImages, 10, 3, 0.5, 5);
		
		totals(nn);
	}
	
	//@Test
	public void dropout() {
		NN nn = new NN(getOpts(), IMGSIZE, 100, 30, 10);
		for(int i=0; i<5; i++) {
			nn.dropout(1);
			nn.SGD(trainingImages, 10, 1, 0.5, 5);
			nn.restore();
			
			nn.dropout(2);
			nn.SGD(trainingImages, 10, 1, 0.5, 5);
			nn.restore();
			
			nn.SGD(trainingImages, 10, 1, 0.5, 5);
		}
		
		totals(nn);
	}
	
	//@Test
	public void augment() {
		System.out.println("Augmenting images...");
		List<Image> augmentedTraining = new ArrayList<>(trainingImages);
		for(Image i : trainingImages) {
			augmentedTraining.add(i.rotate(Math.PI/12));
			
			for(int x=1; x<3; x+=2) for(int y=1; y<3; y+=2) {
				augmentedTraining.add(i.shift(x, y));
				augmentedTraining.add(i.shift(-x, -y));
			}
			
			augmentedTraining.add(i.rotate(-Math.PI/12));
		}

		NN nn = new NN(getOpts(), IMGSIZE, 50, 30, 10);
		nn.SGD(augmentedTraining, 10, 10, 0.5, 5);
		totals(nn);
	}
	
	//@Test
	public void everything() {
		System.out.println("Augmenting images...");
		List<Image> augmentedTraining = new ArrayList<>(trainingImages);
		for(Image i : trainingImages) {
			augmentedTraining.add(i.rotate(Math.PI/12));
			
			for(int x=1; x<3; x+=2) for(int y=1; y<3; y+=2) {
				augmentedTraining.add(i.shift(x, y));
				augmentedTraining.add(i.shift(-x, -y));
			}
			
			augmentedTraining.add(i.rotate(-Math.PI/12));
		}
		
		NN nn = new NN(getOpts(), IMGSIZE, 100, 30, 10);
		
		for(int i=0; i<5; i++) {
			nn.SGD(augmentedTraining, 10, 1, 0.5, 5);
			
			nn.dropout(1);
			nn.SGD(augmentedTraining, 10, 1, 0.5, 0);
			
			nn.invertDropout();
			nn.SGD(augmentedTraining, 10, 1, 0.5, 0);
			
			nn.restore();
		}
		
		nn.getOptions().summarizeSGD = true;
		nn.SGD(trainingImages, 10, 5, 0.25, 3);
		
		totals(nn);
	}
}
