package kricket.neural.util;

import java.util.Set;

/**
 * What a crackhead; who implements their own Matrix class?
 * I actually started with JAMA, but JProfiler showed me that a lot of time
 * was being wasted creating double[] arrays - because JAMA matrices store
 * their data as double[][], every temporary matrix had to initialize all
 * those arrays (java's not the most efficient language when it comes to
 * multi-dim arrays). Since my needs are pretty simple, I rolled my own...
 */
public class Matrix {

	/**
	 * (r,c) = data[cols*r + c]<br>
	 * i.e., data is ordered like:<pre>
	 * 0 1 2
	 * 3 4 5
	 * 6 7 8</pre>
	 */
	public final double[] data;
	/**
	 * Number of rows and columns in this Matrix.
	 */
	public final int rows, cols;
	
	/**
	 * Create an empty Matrix of the given size.
	 * @param r Number of rows
	 * @param c Number of columns
	 */
	public Matrix(int r, int c) {
		this(r,c,new double[r*c]);
	}
	
	/**
	 * Create a column vector from the given data.
	 * @param col The column vector.
	 */
	public Matrix(double[] col) {
		this(col.length, 1, col);
	}
	
	/**
	 * Create a new Matrix of the given size, using the given data.
	 * @param r
	 * @param c
	 * @param data
	 */
	public Matrix(int r, int c, double[] data) {
		rows = r;
		cols = c;
		this.data = data;
	}
	
	/**
	 * Initialize this with random values in (-1, 1).
	 * @param r
	 * @param c
	 * @return
	 */
	public static Matrix random(int r, int c) {
		Matrix m = new Matrix(r, c);
		for(int i=0; i<m.data.length; i++) {
			m.data[i] = Math.random() - Math.random();
		}
		return m;
	}
	
	/**
	 * Get the value at row r, column c.
	 * @param r
	 * @param c
	 * @return
	 */
	public double at(int r, int c) {
		return data[r*cols + c];
	}
	
	/**
	 * Set the value at row r, column c.
	 * @param r
	 * @param c
	 * @param value
	 */
	public void set(int r, int c, double value) {
		data[r*cols+c] = value;
	}
	
	public Matrix plusEquals(Matrix m) {
		/*
		if(data.length != m.data.length)
			throw new IllegalArgumentException("I have " + data.length + " elements, but m has " + m.data.length);
		*/
		for(int i=0; i<data.length; i++)
			data[i] += m.data[i];
		return this;
	}

	public Matrix times(Matrix m) {
		/*
		if(cols != m.rows)
			throw new IllegalArgumentException("Incompatible dimensions: I have " + cols + " cols, but m has " + m.rows + " rows");
		*/
		Matrix p = new Matrix(rows, m.cols);
		for(int r=0; r<p.rows; r++) {
			for(int c=0; c<p.cols; c++) {
				// this.row r . m.col c
				double rc = 0;
				for(int i=0; i<cols; i++) {
					rc += data[r*cols + i] * m.data[i*m.cols + c];
				}
				p.data[p.cols*r + c] = rc;
			}
		}
		return p;
	}
	
	/**
	 * Multiply the transpose of this times m
	 * @param m
	 * @return
	 */
	public Matrix transposeTimes(Matrix m) {
		/*
		if(rows != m.rows)
			throw new IllegalArgumentException("Incompatible dimensions: my transpose has " + rows + " cols, but m has " + m.rows + " rows");
		*/
		Matrix p = new Matrix(cols, m.cols);
		for(int r=0; r<p.rows; r++) {
			for(int c=0; c<p.cols; c++) {
				// this.col r . m.col c
				double rc = 0;
				for(int i=0; i<rows; i++) {
					rc += data[i*cols + r] * m.data[i*m.cols + c];
				}
				p.data[p.cols*r + c] = rc;
			}
		}
		return p;
	}
	
	/**
	 * Multiply this times the transpose of m.
	 * @param m
	 * @return
	 */
	public Matrix timesTranspose(Matrix m) {
		/*
		if(cols != m.cols)
			throw new IllegalArgumentException("Incompatible dimensions: I have " + cols + " cols, but mT has " + m.cols + " rows");
		*/
		Matrix p = new Matrix(rows, m.rows);
		for(int r=0; r<p.rows; r++) {
			for(int c=0; c<p.cols; c++) {
				// this.row r . m.row c
				double rc = 0;
				for(int i=0; i<cols; i++) {
					rc += data[r*cols + i] * m.data[c*m.cols + i];
				}
				p.data[p.cols*r + c] = rc;
			}
		}
		return p;
	}
	
	public Matrix copy() {
		Matrix m = new Matrix(rows, cols);
		for(int i=0; i<data.length; i++) {
			m.data[i] = data[i];
		}
		return m;
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for(int i=0; i<data.length; i++) {
			if(i%cols == 0 && i>0)
				sb.append("\n");
			
			sb.append(" ");
			sb.append(Math.round(data[i]*1000) * 0.001);
		}
		
		return sb.toString();
	}

	public Matrix timesEquals(double d) {
		for(int i=0; i<data.length; i++)
			data[i] *= d;
		return this;
	}
	
	public Matrix dotTimesEquals(Matrix m) {
		/*
		if(cols != m.cols || rows != m.rows)
			throw new IllegalArgumentException("Incompatible dimensions: I am (" + rows + "," + cols + "), m is (" + m.rows + "," + m.cols + ")");
		*/
		for(int i=0; i<data.length; i++)
			data[i] *= m.data[i];
		
		return this;
	}

	public Matrix minus(Matrix m) {
		/*
		if(cols != m.cols || rows != m.rows)
			throw new IllegalArgumentException("Incompatible dimensions: I am (" + rows + "," + cols + "), m is (" + m.rows + "," + m.cols + ")");
		*/
		Matrix result = new Matrix(rows, cols);
		for(int i=0; i<data.length; i++)
			result.data[i] = data[i] - m.data[i];
		return result;
	}

	/**
	 * Return a copy of this Matrix, WITHOUT the given rows.
	 * @param rowsToRemove Indices of the rows to remove.
	 */
	public Matrix withoutRows(Set<Integer> rowsToRemove) {
		Matrix m = new Matrix(rows - rowsToRemove.size(), cols);
		int mr = 0;
		for(int r=0; r<rows; r++) {
			if(rowsToRemove.contains(r))
				continue;
			for(int c=0; c<cols; c++) {
				m.data[mr*cols + c] = data[r*cols + c];
			}
			mr++;
		}
		
		return m;
	}

	/**
	 * Return a copy of this Matrix, WITHOUT the given columns.
	 * @param colsToRemove Indices of the columns to remove.
	 */
	public Matrix withoutColumns(Set<Integer> colsToRemove) {
		Matrix m = new Matrix(rows, cols - colsToRemove.size());
		for(int r=0; r<rows; r++) {
			int mc = 0;
			for(int c=0; c<cols; c++) {
				if(colsToRemove.contains(c))
					continue;
				m.data[r*m.cols + mc] = data[r*cols + c];
				mc++;
			}
		}
		return m;
	}

	/**
	 * Inverse of {@link #withoutRows(Set)}: assume that this Matrix had previously had
	 * removeRows() called. The resulting Matrix was modified, and now we want to re-
	 * integrate the missing rows.
	 * @param m A Matrix containing only the rows that were previously removed.
	 * @param rowsToKeep The indices of the rows that were NOT removed (and will thus
	 * be left as-is).
	 */
	public void restoreRows(Matrix m, Set<Integer> rowsToKeep) {
		int mr = 0;
		for(int r=0; r<rows; r++) {
			if(rowsToKeep.contains(r))
				continue;
			
			for(int c=0; c<cols; c++) {
				set(r,c, m.at(mr, c));
			}
			
			mr++;
		}
	}

	public void restoreColumns(Matrix m, Set<Integer> removedCols) {
		for(int r=0; r<rows; r++) {
			int mc=0;
			for(int c=0; c<cols; c++) {
				if(removedCols.contains(c))
					continue;
				data[r*cols + c] = m.data[r*m.cols + mc];
				mc++;
			}
		}
	}
	
	@Override
	public boolean equals(Object other) {
		if(!(other instanceof Matrix))
			return false;
		
		Matrix m = (Matrix) other;
		if(m.rows != rows || m.cols != cols)
			return false;
		
		for(int i=0; i<data.length; i++)
			if(m.data[i] != data[i])
				return false;
		
		return true;
	}

	/**
	 * Sum up the products of the entries of the sub-matrix starting at (startRow, startCol)
	 * with the entries of the given Matrix.
	 * @param startRow
	 * @param startCol
	 * @param kernel
	 * @return
	 */
	public double subMatrixDot(int startRow, int startCol, Matrix kernel) {
		double result = 0;
		int k = 0;
		for(int r=startRow; r<startRow+kernel.rows; r++) {
			for(int c=startCol; c<startCol+kernel.cols; c++) {
				result += at(r, c) * kernel.data[k++];
			}
		}
		return result;
	}

	/**
	 * Add a submatrix of m (scaled by the given factor) to this.
	 * @param m
	 * @param startRow
	 * @param startCol
	 * @param factor
	 */
	public void plusEqualsSubMatrix(Matrix m, int startRow, int startCol, double factor) {
		for(int r=0; r<rows; r++) for(int c=0; c<cols; c++) {
			data[r*cols+c] += m.at(r+startRow, c+startCol) * factor;
		}
	}
	
	public String draw() {
		double min = Double.MAX_VALUE, max = Double.MIN_VALUE;
		for(int r=0; r<rows; r++) for(int c=0; c<cols; c++) {
			double d = at(r,c);
			if(d < min)
				min = d;
			if(d > max)
				max = d;
		}

		double diff = (max-min)/4;
		double x = min + diff;
		double y = x + diff;
		double z = y + diff;
		
		StringBuilder sb = new StringBuilder();
		for(int r=0; r<rows; r++) {
			for(int c=0; c<cols; c++) {
				double value = at(r,c);
				if(value > z)
					sb.append("@");
				else if(value > y)
					sb.append("o");
				else if(value > x)
					sb.append(".");
				else
					sb.append(" ");
			}
			sb.append("\n");
		}
		return sb.toString();
	}
}
