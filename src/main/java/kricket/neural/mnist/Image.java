package kricket.neural.mnist;

import kricket.neural.util.Datum;
import kricket.neural.util.Matrix;
import kricket.neural.util.Tensor;

public class Image implements Datum {

	public static final int WIDTH = 28, HEIGHT = 28;
	/**
	 * 1 = black, 0 = white
	 */
	private Matrix data;
	private byte answerByte;
	private Matrix answer;

	public Image(double[] image, byte value) {
		data = new Matrix(WIDTH, HEIGHT, image);
		answerByte = value;
		answer = new Matrix(10,1);
		answer.set(answerByte, 0, 1);
	}
	
	public Matrix getData() {
		return data;
	}
	
	public Matrix getAnswer() {
		return answer;
	}
	
	public byte getAnswerAsByte() {
		return answerByte;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(data.draw());
		
		for(int i=0; i<answer.data.length; i++) {
			sb.append((int) answer.data[i]);
			sb.append(" ");
		}
		sb.append(" = ");
		sb.append(answerByte);
		
		return sb.toString();
	}
	
	private double pixel(double dx, double dy) {
		int x = (int) dx, y = (int) dy;
		if(x < 0 || y < 0 || x >= WIDTH || y >= HEIGHT)
			return 0;
		return data.at(y, x);
	}
	
	private void setPixel(int x, int y, double value) {
		data.set(y, x, value);
	}
	
	/**
	 * Get a copy of this image, rotated by the given amount.
	 * @param rad
	 * @return
	 */
	public Image rotate(double rad) {
		Image r = new Image(new double[WIDTH*HEIGHT], answerByte);
		// Rotate BACKWARDS
		double sin = Math.sin(-rad), cos = Math.cos(-rad);
		
		final int W = WIDTH/2, H = HEIGHT/2;
		for(int x=0; x<WIDTH; x++) {
			for(int y=0; y<HEIGHT; y++) {
				// rotate (x,y) by -rad about the middle of the image
				double origX = cos*(x-W) - sin*(y-H) + W;
				double origY = sin*(x-W) + cos*(y-H) + H;
				
				// origX is going to be somewhere between two integral values. We use that
				// to scale the amout of each pixel we find from the original image.
				double leftX = 1. - (origX - ((int)origX));
				double botY = 1. - (origY - ((int)origY));
				
				// So, now we have something like:
				// for X: 0.22 left, 0.78 right
				// for Y: 0.61 bot,  0.39 top
				// We just need to combine them
				double result =
						leftX * botY * pixel(origX, origY) +
						leftX * (1-botY) * pixel(origX, origY+1) +
						(1-leftX) * botY * pixel(origX+1, origY) +
						(1-leftX) * (1-botY) * pixel(origX+1, origY+1);
				r.setPixel(x,y,result);
			}
		}
		
		return r;
	}
	
	/**
	 * Get a copy of this Image, shifted by the given amount.
	 * @param x
	 * @param y
	 * @return
	 */
	public Image shift(int x, int y) {
		Image img = new Image(new double[WIDTH*HEIGHT], answerByte);
		
		for(int r=0; r<HEIGHT; r++) {
			for(int c=0; c<WIDTH; c++) {
				img.setPixel(c, r, pixel(c-x,r+y));
			}
		}
		
		return img;
	}

	@Override
	public Tensor getDataTensor() {
		return new Tensor(data.rows, data.cols, 1, data.data);
	}

	@Override
	public Tensor getAnswerTensor() {
		return new Tensor(answer.rows, answer.cols, 1, answer.data);
	}
}
