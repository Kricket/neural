package kricket.neural.cnn;

import static org.junit.Assert.assertEquals;
import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
import kricket.neural.util.Tensor;
import kricket.neural.util.TensorTest;

import org.junit.Test;

public class MaxPoolingLayerTest {
	
	@Test
	public void outputDimIsCorrect() throws IncompatibleLayerException {
		MaxPoolingLayer mp = new MaxPoolingLayer();
		Dimension outputDim = mp.prepare(new Dimension(3,4,2));
		assertEquals(3, outputDim.rows);
		assertEquals(4, outputDim.columns);
		assertEquals(1, outputDim.depth);
		
		Tensor x = new Tensor(3,4,2);
		Tensor forward = mp.feedForward(x);
		
		assertEquals(1, forward.depth);
		assertEquals(3, forward.rows);
		assertEquals(4, forward.cols);
	}
	
	@Test
	public void outputIsCorrect() throws IncompatibleLayerException {
		MaxPoolingLayer mp = new MaxPoolingLayer();
		mp.prepare(new Dimension(3,3,2));
		
		Tensor x = new Tensor(3,3,2, new double[]{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
				
				9, 8, 7,
				6, 5, 4,
				3, 2, 1
		});
		Tensor forward = mp.feedForward(x);
		
		for(int r=0; r<3; r++) for(int c=0; c<3; c++) {
			double val = r*3 + c + 1;
			if(val < 5)
				val = 10-val;
			assertEquals("row " + r + " col " + c, val, forward.at(r, c, 0), TensorTest.TOLERANCE);
		}
	}
}
