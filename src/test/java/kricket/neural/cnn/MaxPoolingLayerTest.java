package kricket.neural.cnn;

import static org.junit.Assert.assertEquals;
import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
import kricket.neural.util.Matrix;
import kricket.neural.util.MatrixTest;

import org.junit.Test;

public class MaxPoolingLayerTest {
	
	@Test
	public void outputDimIsCorrect() throws IncompatibleLayerException {
		MaxPoolingLayer mp = new MaxPoolingLayer();
		Dimension outputDim = mp.prepare(new Dimension(3,3,2));
		assertEquals(3, outputDim.rows);
		assertEquals(3, outputDim.columns);
		assertEquals(1, outputDim.depth);
		
		Matrix[] x = new Matrix[]{MatrixTest.id(3), new Matrix(3,3)};
		Matrix[] forward = mp.feedForward(x);
		
		assertEquals(1, forward.length);
		assertEquals(3, forward[0].rows);
		assertEquals(3, forward[0].cols);
	}
	
	@Test
	public void outputIsCorrect() throws IncompatibleLayerException {
		MaxPoolingLayer mp = new MaxPoolingLayer();
		mp.prepare(new Dimension(3,3,2));
		
		Matrix id3 = MatrixTest.id(3);
		Matrix m = new Matrix(3,3);
		m.set(1, 1, 2);
		Matrix[] x = new Matrix[]{id3, m};
		Matrix[] forward = mp.feedForward(x);
		
		for(int r=0; r<3; r++) for(int c=0; c<3; c++) {
			if(r==1 && c==1)
				assertEquals(2, forward[0].at(r, c), MatrixTest.TOLERANCE);
			else
				assertEquals(id3.at(r,c), forward[0].at(r, c), MatrixTest.TOLERANCE);
		}
	}
}
