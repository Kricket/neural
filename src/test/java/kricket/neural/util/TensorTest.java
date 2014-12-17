package kricket.neural.util;

import static org.junit.Assert.*;

import org.junit.Test;

public class TensorTest {
	
	public final static double TOLERANCE = 0.00000000001;
	
	/**
	 * Get a square ID Matrix (Tensor of depth 1) of the given size.
	 * @param dim Number of rows and columns in the result.
	 * @return
	 */
	public static Tensor id(int dim) {
		Tensor id = new Tensor(dim, dim, 1);
		for(int i=0; i<dim; i++) {
			id.set(i, i, 0, 1);
		}
		return id;
	}

	@Test
	public void basics() {
		Tensor m = new Tensor(10, 20, 1);
		assertEquals(10, m.rows);
		assertEquals(20, m.cols);
		assertEquals(1,  m.slices);
		assertEquals(200, m.data.length);
	}
	
	@Test
	public void equals() {
		Tensor m = Tensor.random(10, 20, 1);
		Tensor mCopy = m.copy();
		assertEquals(m, mCopy);
		assertFalse(m == mCopy);
		m.data[2] += 3;
		assertNotEquals(m, mCopy);
	}
	
	@Test
	public void plusEquals() {
		Tensor m = Tensor.random(5, 1, 1);
		Tensor mCopy = m.copy();
		m.plusEquals(m);
		for(int i=0; i<m.data.length; i++) {
			assertEquals("index " + i, m.data[i], mCopy.data[i]*2, TOLERANCE);
		}
	}
	
	@Test
	public void timesTensor() {
		Tensor m = Tensor.random(3, 10, 1), id10 = id(10), id3 = id(3);
		Tensor temp = new Tensor(3, 10, 1);
		assertEquals(m.times(id10, temp), m);
		assertEquals(id3.times(m, temp), m);
	}
	
	@Test
	public void timesTransposeTimes() {
		Tensor m = new Tensor(2,3,1), n = new Tensor(3,2,1), id2 = id(2), id3 = id(3);
		int i = 0;
		for(int r=0; r<2; r++) {
			for(int c=0; c<3; c++) {
				m.set(r, c, 0, i);
				n.set(c, r, 0, i);
				i++;
			}
		}
		
		Tensor t3x2 = new Tensor(3,2,1), t2x3 = new Tensor(2,3,1);
		assertEquals(m.transposeTimes(id2,t3x2), n);
		assertEquals(n.transposeTimes(id3,t2x3), m);
		
		assertEquals(id2.timesTranspose(n,t2x3), m);
		assertEquals(id3.timesTranspose(m,t3x2), n);
	}
	
	@Test
	public void timesEquals() {
		final double FACTOR = -1.23;
		Tensor m = Tensor.random(3, 1, 1);
		Tensor n = m.copy();
		n.timesEquals(FACTOR);
		
		for(int i=0; i<m.data.length; i++) {
			assertEquals("data[" + i + "]", m.data[i]*FACTOR, n.data[i], TOLERANCE);
		}
	}
	
	@Test
	public void dotTimesEquals() {
		Tensor m = Tensor.random(2, 3, 1), n = Tensor.random(2, 3, 1);
		Tensor nCopy = n.copy();
		n.dotTimesEquals(m);
		for(int i=0; i<n.data.length; i++) {
			assertEquals("data[" + i + "]", m.data[i]*nCopy.data[i], n.data[i], TOLERANCE);
		}
	}
	
	@Test
	public void minus() {
		Tensor m = Tensor.random(3, 5, 1), n = Tensor.random(3, 5, 1);
		Tensor minus = m.minus(n);
		for(int i=0; i<m.data.length; i++) {
			assertEquals("data[" + i + "]", m.data[i] - n.data[i], minus.data[i], TOLERANCE);
		}
	}
	/*
	@Test
	public void withoutRows() {
		Tensor m = Tensor.random(10,20,1);
		Set<Integer> rows = new HashSet<>(Arrays.asList(1, 3, 5, 7, 9));
		Tensor without = m.withoutRows(rows);
		
		int wr = 0;
		for(int r=0; r<10; r++) {
			if(rows.contains(r))
				continue;
			
			for(int c=0; c<10; c++) {
				assertEquals("row " + r + " col " + c, m.at(r,c), without.at(wr,c), TOLERANCE);
			}
			wr++;
		}
	}
	
	@Test
	public void withoutColumns() {
		Tensor m = Tensor.random(10,20);
		Set<Integer> cols = new HashSet<>(Arrays.asList(1, 3, 5, 7, 9));
		Tensor without = m.withoutColumns(cols);
		
		for(int r=0; r<10; r++) {
			int wc = 0;
			for(int c=0; c<10; c++) {
				if(cols.contains(c))
					continue;
				assertEquals("row " + r + " col " + c, m.at(r,c), without.at(r,wc), TOLERANCE);
				wc++;
			}
		}
	}
	
	@Test
	public void restoreRows() {
		final int COLS = 20;
		Tensor m = new Tensor(10, COLS);
		for(int i=0; i<m.data.length; i++) {
			m.data[i] = 1;
		}
		
		Set<Integer> rows = new HashSet<>(Arrays.asList(1, 3, 5, 7, 9));
		m.restoreRows(new Tensor(rows.size(), COLS), rows);
		
		for(int r=0; r<10; r++) {
			for(int c=0; c<COLS; c++) {
				assertEquals("row " + r + " col " + c,
						m.at(r,c),
						(rows.contains(r) ? 1 : 0),
						TOLERANCE);
			}
		}
	}
	
	@Test
	public void restoreCols() {
		final int ROWS = 20;
		Tensor m = new Tensor(ROWS, 10);
		for(int i=0; i<m.data.length; i++) {
			m.data[i] = 1;
		}
		
		Set<Integer> cols = new HashSet<>(Arrays.asList(1, 3, 5, 7, 9));
		m.restoreColumns(new Tensor(ROWS, cols.size()), cols);
		
		for(int r=0; r<ROWS; r++) {
			for(int c=0; c<10; c++) {
				assertEquals("row " + r + " col " + c,
						m.at(r,c),
						(cols.contains(c) ? 1 : 0),
						TOLERANCE);
			}
		}
	}
	
	@Test
	public void withoutThenRestoreRows() {
		Tensor m = id(3);
		Set<Integer> set = new HashSet<>(Arrays.asList(1));
		Tensor w = m.withoutRows(set);
		m.restoreRows(w, set);
		assertEquals(id(3), m);
	}
	
	@Test
	public void withoutThenRestoreCols() {
		Tensor m = id(3);
		Set<Integer> set = new HashSet<>(Arrays.asList(1));
		Tensor w = m.withoutColumns(set);
		m.restoreColumns(w, set);
		assertEquals(id(3), m);
	}
	*/
	
	@Test
	public void norm() {
		for(int i=1; i<10; i++) {
			Tensor id = id(i);
			assertEquals(Math.sqrt(i), id.norm(), TOLERANCE);
		}
	}
	
	@Test
	public void subTensor_innerProduct() {
		Tensor t = new Tensor(3, 3, 2, new double[]{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
				
				9, 8, 7,
				6, 5, 4,
				3, 2, 1
		});
		
		Tensor id2 = id(2);
		
		double d = new SubTensor(t, 0, 0, 0, 2, 2, 1).innerProduct(id2);
		assertEquals(6, d, TOLERANCE);
		
		d = new SubTensor(t, 1, 1, 1, 2, 2, 1).innerProduct(id2);
		assertEquals(6, d, TOLERANCE);
		
		Tensor id22 = new Tensor(2, 2, 2, new double[] {
				1, 0,
				0, 1,
				
				1, 0,
				0, 1
		});
		
		for(int r=0; r<2; r++) for(int c=0; c<2; c++) {
			d = new SubTensor(t, r, c, 0, 2, 2, 2).innerProduct(id22);
			assertEquals("SubTensor at " + r + ", " + c, 20, d, TOLERANCE);
		}
	}
	
	@Test
	public void subTensor_plusEqualsTimes() {
		Tensor t = new Tensor(3, 3, 2, new double[]{
				1, 1, 1,
				1, 1, 1,
				1, 1, 1,
				
				1, 1, 1,
				1, 1, 1,
				1, 1, 1
		});
		Tensor id3 = id(3);
		
		new SubTensor(t, 0, 0, 0, 3, 3, 1).plusEqualsTimes(id3, -1);
		new SubTensor(t, 0, 0, 1, 3, 3, 1).plusEqualsTimes(id3, -1);
		for(int s=0; s<2; s++) for(int r=0; r<3; r++) for(int c=0; c<3; c++) {
			assertEquals("row " + r + " col " + c + " slice " + s,
					(r == c ? 0 : 1),
					t.at(r, c, s),
					TOLERANCE);
		}
	}
}
