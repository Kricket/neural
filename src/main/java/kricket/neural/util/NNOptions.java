package kricket.neural.util;

import java.util.logging.ConsoleHandler;
import java.util.logging.Formatter;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

/**
 * Simple struct of non-functional options for a NN.
 */
public class NNOptions {
	public NNOptions() {
		log = Logger.getAnonymousLogger();
		log.setUseParentHandlers(false);
		ConsoleHandler ch = new ConsoleHandler();
		log.addHandler(ch);
		ch.setFormatter(new Formatter() {
			@Override
			public String format(LogRecord record) {
				return record.getMessage() + "\n";
			}
		});
	}
	
	public Logger log;
	/**
	 * Whether to log a summary of the SGD run before starting.
	 */
	public boolean summarizeSGD = true;
	/**
	 * Whether to log each epoch when it starts and finishes.
	 */
	public boolean logEpochs = true;
	/**
	 * Whether to calculate the %error after each epoch.
	 */
	public boolean calcErrorsAfterEpochs = true;
	/**
	 * When calculating the %error, use this to log every Datum that was incorrect.
	 */
	public boolean logIncorrectAnswers = false;
	/**
	 * Whether to log the neurons that were dropped out.
	 */
	public boolean logDropout = true;
}
