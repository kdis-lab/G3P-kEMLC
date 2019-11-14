package coeaglet.utils;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import mulan.data.MultiLabelInstances;

/**
 * Class to filter and transform a multi-label dataset given a list (a.k.a. genotype) of active labels
 * 
 * @author Jose M. Moyano
 *
 */
public class DatasetTransformation {
	
	/**
	 *  Filter to remove the non-active labels
	 */
	private Remove filter;
	
	
	/**
	 * Constructor
	 */
	public DatasetTransformation()
	{
		filter = new Remove();
	}
	

	/**
	 * Method to transform a multi-label dataset
	 * 
	 * @param originalDataset Original multi-label dataset to transform
	 * @param genotype List of integers indicating the labels to keep
	 * @return Modified multi-label dataset
	 */
	public MultiLabelInstances transformDataset(MultiLabelInstances originalDataset, List<Integer> genotype)
	{	
		configureFilter(originalDataset, genotype);
		
		MultiLabelInstances modifiedDataset = null;
	    
	    try{
	    	//Instances without the selected labels to remove
	    	Instances modified = Filter.useFilter(originalDataset.getDataSet(), filter);
		    
	    	//Create MultiLabelInstances based on the previous Instances
		    modifiedDataset = originalDataset.reintegrateModifiedDataSet(modified);
	    }
	    catch(Exception e)
	    {
	    	e.printStackTrace();
	    }
	    
	    return modifiedDataset;
	}
	
	/**
	 * Method to transform a single instance of the data, with pre-build filter
	 * 
	 * @param instance Instance of dataset
	 */
	public Instance transformInstance(Instance instance) {
		try {
			filter.input(instance);
			filter.batchFinished();
			return filter.output();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return null;		
	}
	
	
	/**
	 * Configures the filter with the indices of the labels to remove
	 * 
	 * @param originalDataset Original multi-label dataset
	 * @param genotype List of integers indicating the labels to keep
	 */
	private void configureFilter(MultiLabelInstances originalDataset, List<Integer> genotype)
	{
		//Obtain label indices
		int [] labelIndices = originalDataset.getLabelIndices();
		
		//Array with indices of labels to remove
			//Add all labels and remove those to keep
		List<Integer> labelsToRemove = Arrays.stream(originalDataset.getLabelIndices()).boxed().collect(Collectors.toList());
			//I do it in this way because, if not, I should include indexes of all feature attributes

		//Remove from the list those labels to keep
		for(int g : genotype) {
			labelsToRemove.remove(new Integer(labelIndices[g]));
		}
		
		try{
			//Indicate labels to remove in the dataset
			filter.setAttributeIndicesArray(labelsToRemove.stream().mapToInt(i->i).toArray()); //parameter is list transformed to int[] array
			filter.setInvertSelection(false);
			filter.setInputFormat(originalDataset.getDataSet());
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}

}
