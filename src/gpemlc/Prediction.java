package gpemlc;

import java.util.Arrays;

/**
 * Class implementing the predictions of a given classifier
 * 
 * @author Jose M. Moyano
 *
 */
public class Prediction {
	
	/**
	 * Number of instances
	 */
	public int nInstances;
	
	/**
	 * Number of labels
	 */
	public int nLabels;
	
	/**
	 * Prediction as bipartition
	 */
	public byte[][] bip;
	
	/**
	 * Default constructor
	 */
	public Prediction() {
		nInstances = -1;
		nLabels = -1;
		bip = null;
	}
	
	/**
	 * Constructor with parameters
	 * 
	 * @param nInstances Number of instances
	 * @param nLabels Number of labels
	 */
	public Prediction(int nInstances, int nLabels) {
		this.nInstances = nInstances;
		this.nLabels = nLabels;
		this.bip = new byte[nInstances][nLabels];
	}
	
	/**
	 * Constructor with parameter
	 * 
	 * @param prediction Prediction of the classifier as bipartition
	 */
	public Prediction(byte[][] prediction) {
		this.nInstances = prediction.length;
		this.nLabels = prediction[0].length;
		this.bip = prediction.clone();
	}
	
	/**
	 * Add a prediction to the current one
	 * 
	 * @param other Prediction to add to the current one
	 */
	public void addPrediction(Prediction other) {
		if(this.nInstances != other.nInstances || this.nLabels != other.nLabels) {
			System.out.println("The number of instances or labels is not the same in both predictions.");
			System.exit(-1);
		}
		
		for(int i=0; i<nInstances; i++) {
			for(int j=0; j<nLabels; j++) {
				this.bip[i][j] += other.bip[i][j];
			}
		}
	}
	
	/**
	 * Add a prediction to the current one
	 * 
	 * @param otherBip Bipartitio to add to the current one
	 */
	public void addPrediction(byte[][] otherBip) {
		if(this.nInstances != otherBip.length || this.nLabels != otherBip[0].length) {
			System.out.println("The number of instances or labels is not the same in both predictions.");
			System.exit(-1);
		}
		
		for(int i=0; i<nInstances; i++) {
			for(int j=0; j<nLabels; j++) {
				this.bip[i][j] += otherBip[i][j];
			}
		}
	}
	
	/**
	 * Divide the current prediction by the given number of predictions and apply the threshold.
	 * If each prediction is lower than the thresohld, it is negative; and positive otherwise.
	 * 
	 * @param nPreds Number of predictions to divide
	 * @param threshold Threshold
	 */
	public void divideAndThresholdPrediction(int nPreds, double threshold) {
		for(int i=0; i<nInstances; i++) {
			for(int j=0; j<nLabels; j++) {
				if((this.bip[i][j]*1.0)/nPreds >= threshold) {
					this.bip[i][j] = 1;
				}
				else {
					this.bip[i][j] = 0;
				}
			}
		}
	}
	
	@Override
	public String toString() {
		String s = "";
		
		for(int i=0; i<nInstances; i++) {
			s += Arrays.toString(this.bip[i]) + "\n";
		}
	
		return s;
	}
	
}
