package kricket.neural.util;

/**
 * Compact representation of a rank-3 tensor.
 */
public class Tensor {

	/**
	 * The entries of this Tensor.
	 */
	public final double[] data;
	/**
	 * The dimensions of this Tensor.
	 */
	public final int rows, cols, slices;
	
	/**
	 * Get a Tensor filled with random values between -1 and 1.
	 * @param r
	 * @param c
	 * @param s
	 * @return
	 */
	public static Tensor random(int r, int c, int s) {
		Tensor t = new Tensor(r,c,s);
		for(int i=0; i<t.data.length; i++) {
			t.data[i] = Math.random() - Math.random();
		}
		return t;
	}
	
	public Tensor(int r, int c, int s, double[] dat) {
		data = dat;
		rows = r;
		cols = c;
		slices = s;
	}
	
	/**
	 * Create a column vector.
	 * @param vector
	 */
	public Tensor(double[] vector) {
		this(vector.length,1,1,vector);
	}
	
	/**
	 * Create a Tensor with all 0 entries.
	 * @param r
	 * @param c
	 * @param s
	 */
	public Tensor(int r, int c, int s) {
		this(r,c,s,new double[r*c*s]);
	}
	
	public Tensor(Dimension dim) {
		this(dim.rows, dim.columns, dim.depth);
	}

	public Tensor(Dimension dim, double... dat) {
		this(dim.rows, dim.columns, dim.depth, dat);
	}

	public Dimension getDimension() {
		return new Dimension(rows, cols, slices);
	}
	
	private void checkDimensions(Tensor t) {
		if(rows != t.rows || cols != t.cols || slices != t.slices)
			throw new IllegalArgumentException("Incompatible dimensions: I am " + getDimension() + ", t is " + t.getDimension());
	}
	
	/**
	 * Get the index in the {@link #data} array for the given entry.
	 * @param row
	 * @param col
	 * @param slice
	 * @return
	 */
	public int index(int row, int col, int slice) {
		/*
		if(row >= rows || col >= cols || slice >= depth)
			throw new IllegalArgumentException("row " + row + " col " + col + " slice " + slice);
		*/
		return slice*rows*cols + row*cols + col;
	}

	/**
	 * Get the value of an entry.
	 * @param row
	 * @param col
	 * @param slice
	 * @return
	 */
	public double at(int row, int col, int slice) {
		return data[index(row, col, slice)];
	}

	/**
	 * Set the value of an entry.
	 * @param row
	 * @param col
	 * @param slice
	 * @param value
	 */
	public void set(int row, int col, int slice, double value) {
		data[index(row, col, slice)] = value;
	}
	
	/**
	 * Get a new Tensor equal to (this - t).
	 * @param t
	 * @return
	 */
	public Tensor minus(Tensor t) {
		//checkDimensions(t);
		
		Tensor result = new Tensor(rows, cols, slices);
		for(int i=0; i<data.length; i++)
			result.data[i] = data[i] - t.data[i];
		return result;
	}

	/**
	 * Elementwise multiplication with the given tensor.
	 * @param t
	 */
	public void dotTimesEquals(Tensor t) {
		//checkDimensions(t);
		
		for(int i=0; i<data.length; i++) {
			data[i] *= t.data[i];
		}
	}

	/**
	 * Set (this = this + t)
	 * @param t
	 * @return this
	 */
	public Tensor plusEquals(Tensor t) {
		//checkDimensions(t);
		for(int i=0; i<data.length; i++) {
			data[i] += t.data[i];
		}
		return this;
	}
	
	/**
	 * Set this = this*d (for each element)
	 * @param d
	 * @return this
	 */
	public Tensor timesEquals(double d) {
		for(int i=0; i<data.length; i++)
			data[i] *= d;
		return this;
	}
	
	/**
	 * For each slice i, multiply (this[i]) * (the transpose of t[i])
	 * @param t
	 * @param result Storage for the result
	 * @return The given result (for convenience).
	 */
	public Tensor timesTranspose(Tensor t, Tensor result) {
		/*
		if(t.depth != depth)
			throw new IllegalArgumentException("I am " + depth + " deep, but t is " + t.depth);
		if(result.depth != depth)
			throw new IllegalArgumentException("I am " + depth + " deep, but result is " + result.depth);
		if(cols != t.cols)
			throw new IllegalArgumentException("I have " + cols + " cols, but t has " + t.cols);
		if(result.rows != rows || result.cols != t.rows)
			throw new IllegalArgumentException("Incompatible dimensions");
		*/
		
		for(int s=0; s<slices; s++) {
			for(int r=0; r<rows; r++) {
				for(int c=0; c<t.rows; c++) {
					// this.row[r] . t.row[c]
					double rc = 0;
					for(int i=0; i<cols; i++) {
						rc += at(r,i,s) * t.at(c,i,s);
					}
					result.set(r, c, s, rc);
				}
			}
		}
		
		return result;
	}

	/**
	 * For each slice i, multiply (the transpose of this[i]) * (t[i])
	 * @param t
	 * @param result Storage for the result
	 * @return The given result (for convenience).
	 */
	public Tensor transposeTimes(Tensor t, Tensor result) {
		/*
		if(t.depth != depth)
			throw new IllegalArgumentException("I am " + depth + " deep, but t is " + t.depth);
		if(result.depth != depth)
			throw new IllegalArgumentException("I am " + depth + " deep, but result is " + result.depth);
		if(rows != t.rows)
			throw new IllegalArgumentException("I have " + cols + " cols, but t has " + t.cols);
		if(result.rows != cols || result.cols != t.cols)
			throw new IllegalArgumentException("Incompatible dimensions");
		*/
		
		for(int s=0; s<slices; s++) {
			for(int r=0; r<cols; r++) {
				for(int c=0; c<t.cols; c++) {
					// this.col[r] . t.col[c]
					double rc = 0;
					for(int i=0; i<rows; i++) {
						rc += at(i,r,s) * t.at(i,c,s);
					}
					result.set(r, c, s, rc);
				}
			}
		}
		
		return result;
	}

	/**
	 * Get (this * t)
	 * @param t
	 * @param result Storage for the result
	 * @return The given result (for convenience).
	 */
	public Tensor times(Tensor t, Tensor result) {
		/*
		if(t.depth != depth || t.rows != cols)
			throw new IllegalArgumentException("Bad tensor dimension: I am " + getDimension() + ", t is " + t.getDimension());
		if(rows != result.rows || t.cols != result.cols || result.depth != depth)
			throw new IllegalArgumentException("Bad result dimension");
		*/
		for(int s=0; s<slices; s++) {
			for(int r=0; r<rows; r++) {
				for(int c=0; c<t.cols; c++) {
					// this.row[r] . t.col[c]
					double rc = 0;
					for(int i=0; i<cols; i++) {
						rc += at(r,i,s) * t.at(i,c,s);
					}
					result.set(r,c,s,rc);
				}
			}
		}
		
		return result;
	}

	/**
	 * Get a Tensor with the same size and entries as this.
	 * @return
	 */
	public Tensor copy() {
		Tensor t = new Tensor(rows, cols, slices);
		System.arraycopy(data, 0, t.data, 0, data.length);
		return t;
	}

	/**
	 * Get the euclidean norm of this.
	 * @return
	 */
	public double norm() {
		double d = 0;
		for(int i=0; i<data.length; i++)
			d += data[i]*data[i];
		return Math.sqrt(d);

	}
	
	@Override
	public boolean equals(Object o) {
		if(!(o instanceof Tensor))
			return false;
		Tensor t = (Tensor) o;
		
		if(t.rows != rows || t.cols != cols || t.slices != slices)
			return false;
		
		for(int i=0; i<data.length; i++)
			if(data[i] != t.data[i])
				return false;
		
		return true;
	}

	/**
	 * Attempt to pretty-print a Tensor with some ascii art.
	 * @param slice
	 * @return
	 */
	public String draw(int slice) {
		double min = Double.MAX_VALUE, max = Double.MIN_VALUE;
		for(int r=0; r<rows; r++) for(int c=0; c<cols; c++) {
			double d = at(r,c,slice);
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
				double value = at(r,c,slice);
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
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		
		for(int s=0; s<slices; s++) {
			if(s > 0)
				sb.append("\n");
			sb.append("Slice ");
			sb.append(s);
			for(int r=0; r<rows; r++) {
				sb.append("\n");
				for(int c=0; c<cols; c++) {
					sb.append(" ");
					sb.append(String.format("%.3f", at(r,c,s)));
				}
			}
		}
		
		return sb.toString();
	}

	/**
	 * Set this = this + (xs*d)
	 * @param xs
	 * @param d
	 */
	public void plusEqualsTimes(SubTensor xs, double d) {
		/*
		if(rows != xs.rows || cols != xs.cols || slices != xs.slices)
			throw new IllegalArgumentException("Illegal dimensions");
		*/
		for(int s=0; s<slices; s++) for(int r=0; r<rows; r++) for(int c=0; c<cols; c++) {
			data[index(r, c, s)] += xs.at(r, c, s) * d;
		}
	}
}
