package gpemlc;

import java.text.DecimalFormat;

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
	 * Prediction (double to be able to store confidences)
	 */
	public double[][] pred;
	
	/**
	 * Default constructor
	 */
	public Prediction() {
		nInstances = -1;
		nLabels = -1;
		pred = null;
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
		this.pred = new double[nInstances][nLabels];
	}
	
	/**
	 * Constructor with parameter
	 * 
	 * @param prediction Prediction of the classifier (allows confidence values)
	 */
	public Prediction(double[][] prediction) {
		this.nInstances = prediction.length;
		this.nLabels = prediction[0].length;
		this.pred = prediction.clone();
	}
	
	/**
	 * Transform the predictions as confidences into bipartitions given a threshold
	 * 
	 * @param threshold Threshold to determine relevant and irrelevant labels
	 * @return Bipartitions
	 */
	public boolean[][] getBipartition(double threshold) {
		boolean[][] bipartition = new boolean[nInstances][nLabels];
		
		for(int i=0; i<nInstances; i++) {
			for(int j=0; j<nLabels; j++) {
				if(this.pred[i][j] >= threshold) {
					bipartition[i][j] = true;
				}
				else {
					bipartition[i][j] = false;
				}
			}
		}
		
		return bipartition;
	}
	
	/**
	 * Transform the predictions as confidences for a given instance into bipartitions given a threshold
	 * 
	 * @param instance Instance to transform prediction into bipartition
	 * @param threshold Threshold to determine relevant and irrelevant labels
	 * @return Bipartitions
	 */
	public boolean[] getBipartition(int instance, double threshold) {
		boolean[] bipartition = new boolean[nLabels];
		
		for(int j=0; j<nLabels; j++) {
			if(this.pred[instance][j] >= threshold) {
				bipartition[j] = true;
			}
			else {
				bipartition[j] = false;
			}
		}
		
		return bipartition;
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
				this.pred[i][j] += other.pred[i][j];
			}
		}
	}
	
	/**
	 * Add a prediction to the current one
	 * 
	 * @param otherBip Prediction to add to the current one
	 */
	public void addPrediction(double[][] otherPred) {
		if(this.nInstances != otherPred.length || this.nLabels != otherPred[0].length) {
			System.out.println("The number of instances or labels is not the same in both predictions.");
			System.exit(-1);
		}
		
		for(int i=0; i<nInstances; i++) {
			for(int j=0; j<nLabels; j++) {
				this.pred[i][j] += otherPred[i][j];
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
				if((this.pred[i][j])/nPreds >= threshold) {
					this.pred[i][j] = 1;
				}
				else {
					this.pred[i][j] = 0;
				}
			}
		}
	}
	
	/**
	 * Divide the current prediction by the number of total predictions
	 * 
	 * @param nPreds Number of predictions to divide
	 */
	public void divide(int nPreds) {
		for(int i=0; i<nInstances; i++) {
			for(int j=0; j<nLabels; j++) {
				this.pred[i][j] /= nPreds;
			}
		}
	}
	
	@Override
	public String toString() {
		DecimalFormat df = new DecimalFormat("#.###");
		
		String s = "";
		
		for(int i=0; i<nInstances; i++) {
			for(int j=0; j<nLabels; j++) {
				s += df.format(this.pred[i][j]) + ", ";
			}
			s += "\n";
		}
	
		return s;
	}
	
}
