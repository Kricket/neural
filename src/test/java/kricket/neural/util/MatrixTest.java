package kricket.neural.util;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import kricket.neural.util.Matrix;

import org.junit.Test;

public class MatrixTest {
	
	public final static double TOLERANCE = 0.00000000001;
	
	/**
	 * Get the square ID matrix of the given size.
	 * @param dim Number of rows and columns in the result.
	 * @return
	 */
	public static Matrix id(int dim) {
		Matrix id = new Matrix(dim, dim);
		for(int i=0; i<dim; i++) {
			id.set(i, i, 1);
		}
		return id;
	}

	@Test
	public void basics() {
		Matrix m = new Matrix(10, 20);
		assertEquals(10, m.rows);
		assertEquals(20, m.cols);
		assertEquals(200, m.data.length);
	}
	
	@Test
	public void equals() {
		Matrix m = Matrix.random(10, 20);
		Matrix mCopy = m.copy();
		assertEquals(m, mCopy);
		assertFalse(m == mCopy);
		m.data[2] += 3;
		assertNotEquals(m, mCopy);
	}
	
	@Test
	public void plusEquals() {
		Matrix m = Matrix.random(5, 1);
		Matrix mCopy = m.copy();
		m.plusEquals(m);
		for(int i=0; i<m.data.length; i++) {
			assertEquals("index " + i, m.data[i], mCopy.data[i]*2, TOLERANCE);
		}
	}
	
	@Test
	public void timesMatrix() {
		Matrix m = Matrix.random(3, 10), id10 = id(10), id3 = id(3);
		assertEquals(m.times(id10), m);
		assertEquals(id3.times(m), m);
	}
	
	@Test
	public void timesTransposeTimes() {
		Matrix m = new Matrix(2,3), n = new Matrix(3,2), id2 = id(2), id3 = id(3);
		int i = 0;
		for(int r=0; r<2; r++) {
			for(int c=0; c<3; c++) {
				m.set(r, c, i);
				n.set(c, r, i);
				i++;
			}
		}
		
		assertEquals(m.transposeTimes(id2), n);
		assertEquals(n.transposeTimes(id3), m);
		
		assertEquals(id2.timesTranspose(n), m);
		assertEquals(id3.timesTranspose(m), n);
	}
	
	@Test
	public void timesEquals() {
		final double FACTOR = -1.23;
		Matrix m = Matrix.random(3, 1);
		Matrix n = m.copy();
		n.timesEquals(FACTOR);
		
		for(int i=0; i<m.data.length; i++) {
			assertEquals("data[" + i + "]", m.data[i]*FACTOR, n.data[i], TOLERANCE);
		}
	}
	
	@Test
	public void dotTimesEquals() {
		Matrix m = Matrix.random(2, 3), n = Matrix.random(2, 3);
		Matrix nCopy = n.copy();
		n.dotTimesEquals(m);
		for(int i=0; i<n.data.length; i++) {
			assertEquals("data[" + i + "]", m.data[i]*nCopy.data[i], n.data[i], TOLERANCE);
		}
	}
	
	@Test
	public void minus() {
		Matrix m = Matrix.random(3, 5), n = Matrix.random(3, 5);
		Matrix minus = m.minus(n);
		for(int i=0; i<m.data.length; i++) {
			assertEquals("data[" + i + "]", m.data[i] - n.data[i], minus.data[i], TOLERANCE);
		}
	}
	
	@Test
	public void withoutRows() {
		Matrix m = Matrix.random(10,20);
		Set<Integer> rows = new HashSet<>(Arrays.asList(1, 3, 5, 7, 9));
		Matrix without = m.withoutRows(rows);
		
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
		Matrix m = Matrix.random(10,20);
		Set<Integer> cols = new HashSet<>(Arrays.asList(1, 3, 5, 7, 9));
		Matrix without = m.withoutColumns(cols);
		
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
		Matrix m = new Matrix(10, COLS);
		for(int i=0; i<m.data.length; i++) {
			m.data[i] = 1;
		}
		
		Set<Integer> rows = new HashSet<>(Arrays.asList(1, 3, 5, 7, 9));
		m.restoreRows(new Matrix(rows.size(), COLS), rows);
		
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
		Matrix m = new Matrix(ROWS, 10);
		for(int i=0; i<m.data.length; i++) {
			m.data[i] = 1;
		}
		
		Set<Integer> cols = new HashSet<>(Arrays.asList(1, 3, 5, 7, 9));
		m.restoreColumns(new Matrix(ROWS, cols.size()), cols);
		
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
		Matrix m = id(3);
		Set<Integer> set = new HashSet<>(Arrays.asList(1));
		Matrix w = m.withoutRows(set);
		m.restoreRows(w, set);
		assertEquals(id(3), m);
	}
	
	@Test
	public void withoutThenRestoreCols() {
		Matrix m = id(3);
		Set<Integer> set = new HashSet<>(Arrays.asList(1));
		Matrix w = m.withoutColumns(set);
		m.restoreColumns(w, set);
		assertEquals(id(3), m);
	}
	
	@Test
	public void subMatrixDot() {
		Matrix id2 = id(2), id5 = id(5);
		for(int r=0; r<3; r++) for(int c = 0; c<3; c++) {
			Matrix subMatrix = id5.subMatrix(r, c, 2, 2);
			assertEquals("Submatrix at row " + r + ", col " + c,
					(r == c ? 2 : 0),
					subMatrix.dot(id2),
					TOLERANCE);
			assertEquals("Submatrix at row " + r + ", col " + c,
					(r == c ? 2 : 0),
					id2.dot(subMatrix),
					TOLERANCE);
		}
	}
	/*
	@Test
	public void plusEqualsSubMatrix() {
		Matrix id5 = id(5);
		Matrix three2 = id(2).timesEquals(3);
		
		for(int r=0; r<3; r++) {
			int c = r;
			Matrix id2 = id(2);
			id2.plusEqualsSubMatrix(id5, r, c, 2);
			assertEquals("Submatrix at row " + r + ", col " + c,
					three2,
					id2);
		}
	}
	*/
}
