package g3pkemlc;

import java.sql.Timestamp;
import java.util.Comparator;
import java.util.Hashtable;
import java.util.concurrent.locks.ReentrantLock;

import g3pkemlc.utils.TreeUtils;
import g3pkemlc.utils.Utils;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.measure.InformationRetrievalMeasures;
import net.sf.jclec.IFitness;
import net.sf.jclec.IIndividual;
import net.sf.jclec.base.AbstractParallelEvaluator;
import net.sf.jclec.fitness.SimpleValueFitness;
import net.sf.jclec.fitness.ValueFitnessComparator;
import net.sf.jclec.stringtree.StringTreeIndividual;
import net.sf.jclec.util.random.IRandGen;

/**
 * Class implementing the evaluator for StringTreeIndividuals.
 * The tree is reduced, combining predictions in each node, and then final prediction is evaluated.
 * 
 * @author Jose M. Moyano
 *
 */
public class Evaluator extends AbstractParallelEvaluator {

	/**
	 * serialVersionUID
	 */
	private static final long serialVersionUID = 3488129084072488337L;

	/**
	 *  Indicates if fitness is to maximize 
	 */
	private boolean maximize = true;
	
	/**
	 *  Fitness comparator 
	 */
	private Comparator<IFitness> COMPARATOR = new ValueFitnessComparator(!maximize);
	
	/**
	 * Full training dataset
	 */
	MultiLabelInstances fullTrainData;
	
	/**
	 * Utils
	 */
	Utils utils = new Utils();
	
	/**
	 * Table with predictions of each classifier over fullTrainData
	 */
	Hashtable<String, Prediction> tablePredictions = new Hashtable<String, Prediction>();
	
	/**
	 * Indicates if confidences are used instead of bipartitions to combine predictions
	 */
	boolean useConfidences;
	
	/**
	 * Beta for fitness
	 */
	double beta;
	
	/**
	 * Random numbers generator
	 */
	IRandGen randgen;
	
	/**
	 * Locker for critical code blocks in parallel execution
	 */
	ReentrantLock lock = new ReentrantLock(); 
	
	/**
	 * Constructor
	 */
	public Evaluator()
	{
		super();
		utils = new Utils();
	}
	

	@Override
	public Comparator<IFitness> getComparator() {
		return COMPARATOR;
	}
	
	/**
	 * Setter for fullTrainData
	 * 
	 * @param fullTrainData Full training data
	 */
	public void setFullTrainData(MultiLabelInstances fullTrainData) {
		this.fullTrainData = fullTrainData;
	}

	/**
	 * Setter for tablePredictions
	 * 
	 * @param tablePredictions Table with predictions of each classifier over full train data
	 */
	public void setTablePredictions(Hashtable<String, Prediction> tablePredictions) {
		this.tablePredictions = (Hashtable<String, Prediction>) tablePredictions;
	}
	
	/**
	 * Setter for useConfidences
	 * 
	 * @param useConfidences true if confidences are used instead of bipartitions to combine predictions
	 */
	public void setUseConfidences(boolean useConfidences) {
		this.useConfidences = useConfidences;
	}
	
	/**
	 * Setter for beta
	 * 
	 * @param beta beta value for fitness
	 */
	public void setBeta(double beta) {
		this.beta = beta;
	}
	
	/**
	 * Setter for randgen
	 * 
	 * @param randgen Randgen
	 */
	public void setRandgen(IRandGen randgen) {
		this.randgen = randgen;
	}
	
	@Override
	protected void evaluate(IIndividual ind) 
	{
		//Lock here the use of randgen to generate the unique key for the individual
		lock.lock();
		String key = Long.toString(new Timestamp(System.currentTimeMillis()).getTime());
		int r = randgen.choose(3, 10);
		for(int i=0; i<r; i++) {
			key += (char)randgen.choose(48, 123);
		}
		lock.unlock();
		
		String gen = ((StringTreeIndividual)ind).getGenotype();

		double fitness = 0.0;
		
		//Get final predictions by reducing the tree
		Prediction pred = TreeUtils.reduce(gen, key, tablePredictions, fullTrainData.getNumInstances(), useConfidences);

		//If the tree does not cover all the labels, the fitness is negative
		if(pred.labelIndices.size() != fullTrainData.getNumLabels()) {
			//The fitness is lower (worse) as it cover less number of labels
			//	In case of hipothetically selecting two negative fitness individuals in tournament, the one that cover more labels is selected
			fitness = ((pred.labelIndices.size()*1.0) / fullTrainData.getNumLabels()) - 1;
		}
		else {
			//Calculate fitness (with ExF and MaF) with the reduced predictions
			fitness = beta*exF(pred, fullTrainData) + (1-beta)*maF(pred, fullTrainData);
		}
		
		//Set individual fitness
		ind.setFitness(new SimpleValueFitness(fitness));
	}
	
	/**
	 * Calculate the Macro-averaged FMeasure (MaF) for given prediction of all instances and ground truth
	 * 
	 * @param pred Predictions
	 * @param mlData Multi-label data
	 * @return Macro-averaged FMeasure
	 */
	protected double maF(Prediction pred, MultiLabelInstances mlData) {
		int[] labelIndices = mlData.getLabelIndices();
		boolean [] ground = new boolean[mlData.getNumLabels()];
		
		int[] tp = new int[mlData.getNumLabels()];
		int[] fp = new int[mlData.getNumLabels()];
		int[] tn = new int[mlData.getNumLabels()];
		int[] fn = new int[mlData.getNumLabels()];
		
		double maF = 0.0;
		
		for(int i=0; i<mlData.getNumInstances(); i++) {
			boolean[] bip = utils.confidenceToBipartition(pred.pred[i], 0.5);
			for(int j=0; j<mlData.getNumLabels(); j++) {
				//Get ground truth for all labels in i-th instance
				if(mlData.getDataSet().get(i).value(labelIndices[j]) >= 0.5) {
					ground[j] = true;
					if(bip[j]) {
						tp[j]++;
					}
					else {
						fn[j]++;
					}
				}
				else {
					ground[j] = false;
					if(bip[j]) {
						fp[j]++;
					}
					else {
						tn[j]++;
					}
				}
			}
		}
		
		for(int j=0; j<mlData.getNumLabels(); j++) {
			maF += InformationRetrievalMeasures.fMeasure(tp[j], fp[j], fn[j], 1);
		}
		maF /= mlData.getNumLabels();


		return maF;
	}
	
	/**
	 * Calculate the Example-FMeasure (ExF) for given prediction of all instances and ground truth
	 * 
	 * @param pred Predictions
	 * @param mlData Multi-label data
	 * @return Example-based FMeasure
	 */
	protected double exF(Prediction pred, MultiLabelInstances mlData) {
		int[] labelIndices = mlData.getLabelIndices();
		boolean [] ground = new boolean[mlData.getNumLabels()];
		
		double exF = 0.0;
		
		for(int i=0; i<mlData.getNumInstances(); i++) {
			for(int j=0; j<mlData.getNumLabels(); j++) {
				//Get boolean ground truth for all labels in i-th instance
				if(mlData.getDataSet().get(i).value(labelIndices[j]) >= 0.5) {
					ground[j] = true;
				}
				else {
					ground[j] = false;
				}
			}
			
			//Calculate exF for prediction (as boolean bipartition) and ground truth of i-th instance
			exF += exFInstance(utils.confidenceToBipartition(pred.pred[i], 0.5), ground);
		}
		
		//Divide the exF by the number of instances and return it
		return exF/mlData.getNumInstances();
	}
	
	
	/**
	 * Calculate the Example-FMeasure (ExF) for given prediction and ground truth for one instance
	 * 
	 * @param pred Prediction of one instance
	 * @param ground Ground truth
	 * @return FMeasure for the instance
	 */
	protected double exFInstance(boolean[] pred, boolean[] ground) {
		int tp=0, fp=0, fn=0;
		
		for(int i=0; i<pred.length; i++) {
			if(pred[i] && ground[i]) {
				tp++;
			}
			else if(pred[i] && !ground[i]) {
				fp++;
			}
			else if(!pred[i] && ground[i]) {
				fn++;
			}
		}
		
		return InformationRetrievalMeasures.fMeasure(tp, fp, fn, 1.0);
	}	
}
