package net.sf.jclec.stringtree;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import net.sf.jclec.ISpecies;
import net.sf.jclec.binarray.BinArrayIndividual;
import net.sf.jclec.binarray.BinArrayIndividualSpecies;

/**
 * Abstract implementation for IBinArraySpecies.
 * 
 * This class  contains a byte array  that contains the genotype schema for all
 * represented individuals. This schema can be set in a subclass of this or can 
 * be calculated from other problem information.
 * 
 * @author Sebastian Ventura
 * 
 * @see BinArrayIndividualSpecies
 */

@SuppressWarnings("serial")
public abstract class StringTreeSpecies implements ISpecies
{
	/////////////////////////////////////////////////////////////////
	// ----------------------------------------------------Properties
	/////////////////////////////////////////////////////////////////
	
	/** Genotype schema */
	
	protected String genotypeSchema;
	
	/////////////////////////////////////////////////////////////////
	// ------------------------------------------------- Constructors
	/////////////////////////////////////////////////////////////////
	
	/**
	 * Empty constructor
	 */
	
	public StringTreeSpecies() 
	{
		super();
	}

	/////////////////////////////////////////////////////////////////
	// ----------------------------------------------- Public methods
	/////////////////////////////////////////////////////////////////
	
	// Factory method
	
	/**
	 * Factory method.
	 * 
	 * @param genotype Individual genotype.
	 * 
	 * @return A new instance of represented class
	 */
	
	public abstract StringTreeIndividual createIndividual(String genotype);	

	// Genotype information

	/**
	 * Informs about individual genotype length.
	 * 
	 * @return getGenotypeSchema().length
	 */
	
	public int getGenotypeLength() 
	{
		Pattern p = Pattern.compile("\\d+"); // "\d" is for digits in regex
		Matcher m = p.matcher(genotypeSchema);
		int count = 0;
		while(m.find()){
			count++;
		}
		
		return count;
	}

	/**
	 * @return This genotype schema
	 */
	
	public String getGenotypeSchema() 
	{
		return genotypeSchema;
	}
}
