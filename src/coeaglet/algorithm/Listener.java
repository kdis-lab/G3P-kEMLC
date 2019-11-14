package coeaglet.algorithm;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.configuration.Configuration;
import org.apache.commons.lang.StringUtils;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MulanEnsembleEvaluator;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.Coverage;
import mulan.evaluation.measure.ErrorSetSize;
import mulan.evaluation.measure.ExampleBasedAccuracy;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.ExampleBasedPrecision;
import mulan.evaluation.measure.ExampleBasedRecall;
import mulan.evaluation.measure.ExampleBasedSpecificity;
import mulan.evaluation.measure.GeometricMeanAveragePrecision;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.HierarchicalLoss;
import mulan.evaluation.measure.IsError;
import mulan.evaluation.measure.MacroFMeasure;
import mulan.evaluation.measure.MacroPrecision;
import mulan.evaluation.measure.MacroRecall;
import mulan.evaluation.measure.MacroSpecificity;
import mulan.evaluation.measure.MeanAveragePrecision;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.MicroAUC;
import mulan.evaluation.measure.MicroFMeasure;
import mulan.evaluation.measure.MicroPrecision;
import mulan.evaluation.measure.MicroRecall;
import mulan.evaluation.measure.MicroSpecificity;
import mulan.evaluation.measure.OneError;
import mulan.evaluation.measure.RankingLoss;
import mulan.evaluation.measure.SubsetAccuracy;
import net.sf.jclec.AlgorithmEvent;
import net.sf.jclec.IAlgorithm;
import net.sf.jclec.IAlgorithmListener;
import net.sf.jclec.IConfigure;
import net.sf.jclec.IIndividual;
import net.sf.jclec.algorithm.MultiPopulationAlgorithm;
import net.sf.jclec.fitness.SimpleValueFitness;
import net.sf.jclec.listind.MultipListIndividual;

/**
 * Class implementing the listener of the algorithm
 * 
 * @author Jose M. Moyano
 *
 */
public class Listener implements IAlgorithmListener, IConfigure {

	/**
	 * Serialization constant
	 */
	private static final long serialVersionUID = -2865418936116733852L;
	
	
	/** 
	 * Report directory name
	 */	
	protected String reportDirName;
	
	/** 
	 * Global report name 
	 */
	protected String globalReportName;
		
	/** 
	 * Report frequency 
	 */
	protected  int reportFrequency;
	
	/** 
	 * Init time
	 */	
	protected long initTime;
	
	/** 
	 * End time
	 */	
	protected long endTime;

	/** 
	 * Report directory 
	 */	
	protected File reportDirectory;
	
	/**
	 * Filename for iteration report
	 */
	String iterEnsembleFilename = "iterEnsemble.rep";

	
	/**
	 * Constructor
	 */
	public Listener()
	{
		super();
	}
	
	
	/**
	 * Get the report directory name
	 * 
	 * @return report directory name
	 */
	public final String getReportDirName() 
	{
		return reportDirName;
	}
	
	/**
	 * Set the report directory name
	 * 
	 * @param reportDirName directory name
	 */
	public final void setReportDirName(String reportDirName) 
	{
		this.reportDirName = reportDirName;
	}
	
	/**
	 * Get the global report name
	 * 
	 * @return global report name
	 */
	public final String getGlobalReportName() 
	{
		return globalReportName;
	}
	
	/**
	 * Set the global report name
	 * 
	 * @param globalReportName report name
	 */
	public final void setGlobalReportName(String globalReportName) 
	{
		this.globalReportName = globalReportName;
	}

	/**
	 * Get the report frequency
	 * 
	 * @return report frequency
	 */
	public final int getReportFrequency() 
	{
		return reportFrequency;
	}

	/**
	 * Set the report frequency
	 * 
	 * @param reportFrequency frequency
	 */
	public final void setReportFrequency(int reportFrequency) 
	{
		this.reportFrequency = reportFrequency;
	}
	

	@Override
	public void algorithmStarted(AlgorithmEvent event) {
		initTime = System.currentTimeMillis();
		
		Date now = new Date();
		String date = new SimpleDateFormat("yyyy.MM.dd'_'HH.mm.ss.SS").format(now);
		
		// Init report directory
		reportDirectory = new File(reportDirName + "_" + date);
		if (! reportDirectory.mkdir())
			throw new RuntimeException("Error creating report directory");
		
		// Do report
		doIterationReport(event.getAlgorithm());
	}
	
	@Override
	public void algorithmFinished(AlgorithmEvent event) {
		endTime = System.currentTimeMillis();
		doDataReport((Alg) event.getAlgorithm());
		doClassificationReport((Alg) event.getAlgorithm());
		
		closeReportFiles((Alg) event.getAlgorithm());
	}
	
	@Override
	public void iterationCompleted(AlgorithmEvent event) {
		doIterationReport(event.getAlgorithm());
	}
	
	@Override
	public void configure(Configuration settings) {
		// Get report-dir-name
		String reportDirName = settings.getString("report-dir-name", "report");
		// Set reportDirName 
		setReportDirName(reportDirName);
		// Get global-report-name
		String globalReportName = settings.getString("global-report-name", "global-report");
		// Set globalReportName 
		setGlobalReportName(globalReportName);
		// Get report-frequency
		int reportFrequency = settings.getInt("report-frequency", 1);
		// Set reportFrequency
		setReportFrequency(reportFrequency);
	}

	@Override
	public void algorithmTerminated(AlgorithmEvent arg0) {
		//Do nothing
	}

	/**
	 * Make a report with individuals and their fitness
	 * 
	 * @param algorithm Algorithm
	 */
	protected void doIterationReport(IAlgorithm algorithm)
	{		
		// Actual generation
		int generation = ((MultiPopulationAlgorithm) algorithm).getGeneration();
					
		if (generation % reportFrequency == 0) 
		{
			int numSubpop = ((MultiPopulationAlgorithm)algorithm).getNumSubpop();
			
			// Population individuals
			List<List<IIndividual>> inds = ((MultiPopulationAlgorithm) algorithm).getMultiInhabitants();
			
			String reportFilename = String.format("Iteration_%d.rep", generation);
				
			try {
				// Report file
				File reportFile = new File(reportDirectory, reportFilename);
				File iterEnsembleFile = new File("reports", iterEnsembleFilename);
				File[] bestIndFile = new File[numSubpop];
				File[] avgIndFile = new File[numSubpop];
				
				for(int p=0; p<numSubpop; p++) {
					bestIndFile[p] = new File("reports", "bestInd_" + p + ".rep");
					avgIndFile[p] = new File("reports", "avgInd_" + p + ".rep");
				}
					
				// Report writer
				FileWriter reportWriter = null;
				FileWriter iterEnsembleWriter = null;
				FileWriter[] bestIndWriter = new FileWriter[numSubpop];
				FileWriter[] avgIndWriter = new FileWriter[numSubpop];
					
				try {
					reportFile.createNewFile();
					reportWriter = new FileWriter (reportFile);
					iterEnsembleWriter = new FileWriter(iterEnsembleFile, true);
					
					for(int p=0; p<numSubpop; p++) {
						bestIndWriter[p] = new FileWriter(bestIndFile[p], true);
						avgIndWriter[p] = new FileWriter(avgIndFile[p], true);
					}
				}
				catch(IOException e3){
					e3.printStackTrace();
				}
					
				StringBuffer buffer = new StringBuffer();
					
				// Prints individuals
				for(int i=0; i<inds.get(0).size(); i++) {
					for(int p=0; p<numSubpop; p++) {
						List<IIndividual> currInds = ((Alg)algorithm).bettersSelector.select(inds.get(p), inds.get(p).size());
						buffer.append((MultipListIndividual)currInds.get(i) + "; " + ((SimpleValueFitness)currInds.get(i).getFitness()).getValue() + "; "); 
					}
					buffer.append(System.getProperty("line.separator"));
				}

				reportWriter.append(buffer.toString());
				reportWriter.close();
				
				List<MultipListIndividual> bests = ((Alg)algorithm).bestIndividuals();
				double [] avgFitness = ((Alg)algorithm).avgFitnessSubpopulation();
				
				for(int p=0; p<numSubpop; p++) {
					bestIndWriter[p].append(((SimpleValueFitness)bests.get(p).getFitness()).getValue() + "; ");
					bestIndWriter[p].close();
					
					avgIndWriter[p].append(avgFitness[p] + "; ");
					avgIndWriter[p].close();
				}
				
				iterEnsembleWriter.append(((Alg)algorithm).getCurrentEnsembleFitness() + "; ");
				iterEnsembleWriter.close();
			} 
			catch (IOException e) {
				throw new RuntimeException("Error writing report file");
			}
		}
	}
	
	protected void closeReportFiles(Alg algorithm) {
		try {
			int numSubpop = algorithm.getNumSubpop();
			
			File[] bestIndFile = new File[numSubpop];
			File[] avgIndFile = new File[numSubpop];
			
			for(int p=0; p<numSubpop; p++) {
				bestIndFile[p] = new File("reports", "bestInd_" + p + ".rep");
				avgIndFile[p] = new File("reports", "avgInd_" + p + ".rep");
			}

			File iterEnsembleFile = new File("reports", iterEnsembleFilename);
			
			FileWriter[] bestIndWriter = new FileWriter[numSubpop];
			FileWriter[] avgIndWriter = new FileWriter[numSubpop];
			FileWriter iterEnsembleWriter = null;
				
			iterEnsembleWriter = new FileWriter(iterEnsembleFile, true);
			iterEnsembleWriter.append(System.getProperty("line.separator"));
			iterEnsembleWriter.close();
				
			for(int p=0; p<numSubpop; p++) {
				bestIndWriter[p] = new FileWriter(bestIndFile[p], true);
				avgIndWriter[p] = new FileWriter(avgIndFile[p], true);
				
				bestIndWriter[p].append(System.getProperty("line.separator"));
				bestIndWriter[p].close();
				avgIndWriter[p].append(System.getProperty("line.separator"));
				avgIndWriter[p].close();
			}
		} 
		catch (IOException e) {
			throw new RuntimeException("Error writing report file");
		}
	}
	
	
	/**
	 * Make the data report over the train and test datasets
	 * 
	 * @param algorithm Algorithm
	 */
    protected void doDataReport(Alg algorithm)
	{   	
    	// Test report name
		String testReportFilename = "TestDataReport.txt";
		// Train report name
		String trainReportFilename = "TrainDataReport.txt";
		// Test Report file
		File testReportFile = new File(reportDirectory, testReportFilename);
		// Train Report file
		File trainReportFile = new File(reportDirectory, trainReportFilename);
		
		classify(algorithm.getFullTrainData(), algorithm.getEnsemble(), trainReportFile);
		classify(algorithm.getTestData(), algorithm.getEnsemble(), testReportFile);
	}
    
    /**
     * Classify a multi-label dataset with the ensemble, and write the results
     * 
     * @param mldata Multi-label dataset
     * @param classifier Ensemble classifier
     * @param file File to write the predictions
     */
    protected void classify(MultiLabelInstances mldata, Ensemble classifier, File file)
    {
		int[][] predicted = classifier.classify(mldata);
		int numberLabels = mldata.getNumLabels();
		
		try {
    		file.createNewFile();
        	FileWriter fw = new FileWriter(file);
        	fw.write(StringUtils.leftPad("PREDICTED", numberLabels*2-1, " ") + " \t" + StringUtils.leftPad("ACTUAL", numberLabels*2-1, " ") + System.getProperty("line.separator"));
    		
    		for(int i = 0; i < predicted.length; i++)
        	{
    			for(int j = 0; j < numberLabels; j++)
    				fw.write(predicted[i][j] + " ");
    			fw.write("\t");
    			
    			for(int j = 0; j < numberLabels; j++)
    				fw.write((int) mldata.getDataSet().get(i).value(mldata.getDataSet().numAttributes() - numberLabels + j) + " ");
    			fw.write(System.getProperty("line.separator"));
        	}

			fw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
    }
    
    /**
 	 * Make the classifier report over the train and test datasets
 	 * 
 	 * @param algorithm Algorithm
 	 */
    protected void doClassificationReport(Alg algorithm)
    {
    	// Test report name
    	String testReportFilename = "TestClassificationReport.txt";
        // Train report name
    	String trainReportFilename = "TrainClassificationReport.txt";
           	
        MultiLabelInstances datasetTrain = algorithm.getFullTrainData();
        MultiLabelInstances datasetTest = algorithm.getTestData();        
        
        //Build the classifier ENSURE IT WAS INICIALIZED WITH THE GENOTYPE!!!
        Ensemble classifier = algorithm.getEnsemble();
    	try {
    		 classifier.build(datasetTrain);
    	}catch (Exception e1) {
    		 e1.printStackTrace();
    	}   	
    	
		// Check if the report directory name is in a file
		String aux = "";
		if(getReportDirName().split("/").length>1)
			aux = getReportDirName().split("/")[0]+"/";
		else
			aux = "./";
		
        // Global report for train
        String nameFileTrain = aux +getGlobalReportName() + "-train.txt";
    				
    	// Global report for test
        String nameFileTest = aux +getGlobalReportName() + "-test.txt";

        
       doReport(trainReportFilename, nameFileTrain, datasetTrain, algorithm,  classifier);
       doReport(testReportFilename, nameFileTest, datasetTest, algorithm, classifier);
    }
    
    /**
     * Do a report
     * 
     * @param reportFilename Report filename
     * @param globalReportFilename Global report filename
     * @param dataset Multi-label dataset
     * @param algorithm Evolutionary algorithm
     * @param classifier Ensemble classifier
     */
    private void doReport(String reportFilename, String globalReportFilename, MultiLabelInstances dataset, Alg algorithm,  Ensemble classifier)
    {
    	//Report file
    	File reportFile = new File(reportDirectory, reportFilename);
    	//File writer
    	FileWriter file = null;
    	
    	DecimalFormat df4 = new DecimalFormat("0.0000");
    	
    	double evaluateTimeInit,  evaluateTimeEnd;
    	evaluateTimeInit = System.currentTimeMillis();
    	//Evaluating the classifier with the datasetTrain and dataSetTest
    	MulanEnsembleEvaluator eval = new MulanEnsembleEvaluator();
        Evaluation results = null;    			
    	try{
    		  List<Measure> measures = new ArrayList<Measure>();  	       
  	       	  measures = prepareMeasures(classifier, dataset);
    		  results = eval.evaluate(classifier, dataset, measures);
    		 
    	} catch (IllegalArgumentException e1) {
    		  e1.printStackTrace();
    	} catch (Exception e1) {
    		  e1.printStackTrace();
    	}
    	evaluateTimeEnd = System.currentTimeMillis();
    	double evaluateTime = evaluateTimeEnd - evaluateTimeInit;
    	
    	
    	try {
			reportFile.createNewFile();
			file = new FileWriter (reportFile);
			
			file.write("Relation: " + dataset.getDataSet().relationName());
			file.write("\nNumber of attributes: " + (dataset.getDataSet().numAttributes() - dataset.getNumLabels()));
			file.write("\nNumber of labels: " + dataset.getNumLabels());
			file.write("\nRun Time (s): " + (((double) (endTime-initTime)) / 1000.0));
//			file.write("\nEvaluation Time (s): " + (algorithm.getEvaluator().getEvaluationTime() / 1000.0));
			file.write(System.getProperty("line.separator"));
			
			String cab="Dataset, ";
			for (Measure m : results.getMeasures())
		    { 
				file.write("\n"+m.getName()+": "+df4.format(m.getValue()));
				cab=cab+m.getName()+", ";				
		    }
			cab=cab+"number of evaluations, execution time, evaluation time, nClassifiers\n";
			file.write(System.getProperty("line.separator"));
			
			// Global report
			File fileTrain = new File(globalReportFilename);
			BufferedWriter bw;
			
			// If the global report exists
			if(fileTrain.exists())
			{
				bw = new BufferedWriter (new FileWriter(globalReportFilename,true));
				bw.write(System.getProperty("line.separator"));
			}
			else
			{
				bw = new BufferedWriter (new FileWriter(globalReportFilename));
				bw.write(cab);
			}
			
			bw.write(dataset.getDataSet().relationName() + ",");			
			for (Measure m : results.getMeasures())
		    { 	
				bw.write(m.getValue() + ",");
		    }
			
			bw.write(algorithm.getEvaluator().getNumberOfEvaluations() + ",");
//			bw.write((algorithm.getEvaluator().getEvaluationTime() / 1000.0) + ",");
			bw.write((((double)(endTime-initTime)) / 1000.0) + ",");
			bw.write(((evaluateTime) / 1000.0) + ",");
			//bw.write(algorithm.getNumberOfEvaluatedIndividuals() + ", ");
			//bw.write(algorithm.getEnsemble().getNumClassifiers() + ", ");
			
			file.write(System.getProperty("line.separator") + "Ensemble of classifiers" + System.getProperty("line.separator"));
			file.write(classifier.toString());
			
			// Close the files
			bw.close();
			file.close();			
    	} 
		catch (IOException e) 
		{
			e.printStackTrace();
		}
    	
    }     
    
    /**
     * Prepare the measures to evaluate
     * 
     * @param learner Multi-label learner
     * @param mlTestData Multi-label data to evaluate
     * 
     * @return List with the measures
     */
    private List<Measure> prepareMeasures(MultiLabelLearner learner,
            MultiLabelInstances mlTestData) {
        List<Measure> measures = new ArrayList<Measure>();

        MultiLabelOutput prediction;
        try {
            prediction = learner.makePrediction(mlTestData.getDataSet().instance(0));
            int numOfLabels = mlTestData.getNumLabels();
            
            // add bipartition-based measures if applicable
            if (prediction.hasBipartition()) {
                // add example-based measures
                measures.add(new HammingLoss());
                measures.add(new SubsetAccuracy());
                measures.add(new ExampleBasedPrecision());
                measures.add(new ExampleBasedRecall());
                measures.add(new ExampleBasedFMeasure());
                measures.add(new ExampleBasedAccuracy());
                measures.add(new ExampleBasedSpecificity());
                // add label-based measures
                measures.add(new MicroPrecision(numOfLabels));
                measures.add(new MicroRecall(numOfLabels));
                measures.add(new MicroFMeasure(numOfLabels));
                measures.add(new MicroSpecificity(numOfLabels));
                measures.add(new MacroPrecision(numOfLabels));
                measures.add(new MacroRecall(numOfLabels));
                measures.add(new MacroFMeasure(numOfLabels));
                measures.add(new MacroSpecificity(numOfLabels));
            }
            // add ranking-based measures if applicable
            if (prediction.hasRanking()) {
                // add ranking based measures
                measures.add(new AveragePrecision());
                measures.add(new Coverage());
                measures.add(new OneError());
                measures.add(new IsError());
                measures.add(new ErrorSetSize());
                measures.add(new RankingLoss());
            }
            // add confidence measures if applicable
            if (prediction.hasConfidences()) {
                measures.add(new MeanAveragePrecision(numOfLabels));
                measures.add(new GeometricMeanAveragePrecision(numOfLabels));
               // measures.add(new MeanAverageInterpolatedPrecision(numOfLabels, 10));
               // measures.add(new GeometricMeanAverageInterpolatedPrecision(numOfLabels, 10));
                measures.add(new MicroAUC(numOfLabels));
                //measures.add(new MacroAUC(numOfLabels));
               // measures.add(new LogLoss());
            }
            // add hierarchical measures if applicable
            if (mlTestData.getLabelsMetaData().isHierarchy()) {
                measures.add(new HierarchicalLoss(mlTestData));
            }
        } catch (Exception ex) {
            Logger.getLogger(Evaluator.class.getName()).log(Level.SEVERE, null, ex);
        }

        return measures;
    }

}
