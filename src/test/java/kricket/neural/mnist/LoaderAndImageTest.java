package kricket.neural.mnist;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.List;

import kricket.neural.mnist.Image;
import kricket.neural.mnist.Loader;

import org.junit.Test;

public class LoaderAndImageTest {
	@Test
	public void basic() throws IOException {
		byte[] labels = Loader.loadLabels(Loader.LABELS_TRAINING_FILE);
		assertEquals(60000, labels.length);
		List<Image> images = Loader.loadImages(Loader.IMAGES_TRAINING_FILE, Loader.LABELS_TRAINING_FILE);
		assertEquals(60000, images.size());
		/*
		for(int i=0; i<5; i++)
			System.out.println(images.get(i).shift(10, 10));
			*/
	}
}
