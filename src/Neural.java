import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class Neural 
{
	/*************************************************************************/

	//Record class (inner class)
    private class Record 
    {
        private int[][] attributes;       //attributes of record      
        private double[] className;       //class of record (array now)

        //Constructor of Record
        private Record(int[][] attributes, double classValue)
        {
            this.attributes = attributes;    

           
            this.className = new double[1];
            this.className[0] = classValue;
        }
    }
    
    /*************************************************************************/
    
    private ArrayList<Record> records;   
    private int numberRecords;           
         
    private int numberInputs;            //16 for 16x16
    private int numberOutputs;           
    private int numberMiddle;            
    private int numberIterations;        
    private double rate;      
    
    private int numberError;
    private double errorRate;

    private int totalInputs;             //256 for 16x16

    private double[] input;              
    private double[] middle;             
    private double[] output;             

    private double[] errorMiddle;        
    private double[] errorOut;           
    private double[] thetaMiddle;        
    private double[] thetaOut;           

    private double[][] matrixMiddle;     
    private double[][] matrixOut;     
    private int seed;

    /*************************************************************************/
    
    //Constructor of neural network
    public Neural()
    {
    	numberError =0;
        numberRecords = 0;       
        numberInputs = 0;
        numberOutputs = 0;
        numberMiddle = 0;        
        numberIterations = 0;
        rate = 0;

        records = null;          
        input = null;            
        middle = null;
        output = null;
        errorMiddle = null;      
        errorOut = null;
        thetaMiddle = null;      
        thetaOut = null;
        matrixMiddle = null;     
        matrixOut = null;
    }
    
    /*************************************************************************/

    //Method loads training records from training file
    public void loadTrainingFile(String trainingFile) throws IOException
    {
    	Scanner inFile = new Scanner(new File(trainingFile));
    	
    	numberRecords = inFile.nextInt();
    	numberInputs = inFile.nextInt();     // 16 (rows & columns)
    	numberOutputs = inFile.nextInt();    // 1

       
        totalInputs = numberInputs * numberInputs;   // 256
    	
    	records = new ArrayList<Record>();
    	
    	for(int x=0; x<numberRecords; x++)
    	{
        	int fileInput[][] = new int[numberInputs][numberInputs];
        	
        	for(int i=0; i<numberInputs ; i++)
        	{
        		for(int j=0; j<numberInputs ; j++)
        		{
        			fileInput[i][j] = inFile.nextInt();
        		}
        	}
    		
        	int className = inFile.nextInt();
    		
        	double classNum = 0;
        	
        	if(className == 0)
        		classNum =0;
        	else if(className == 1)
        		classNum = 0.5;
        	else if(className == 2)
        		classNum = 1.0;
        	
        	Record oneRecord = new Record(fileInput,classNum);
        	
        	records.add(oneRecord);
    	}
    	
    	inFile.close();
    }
    
    /*************************************************************************/
    
    //Method sets parameters of neural network
    public void setParameters(int numberMiddle, int numberIterations, double rate, int seed)
    {
        this.numberMiddle = numberMiddle;
        this.numberIterations = numberIterations;
        this.rate = rate;

        this.seed = seed;
        
        //create a random number object
        Random rand = new Random(seed);

        
        input = new double[totalInputs];

        middle = new double[numberMiddle];
        output = new double[numberOutputs];

        errorMiddle = new double[numberMiddle];
        errorOut = new double[numberOutputs];

        thetaMiddle = new double[numberMiddle];
        for (int i = 0; i < numberMiddle; i++)
            thetaMiddle[i] = 2*rand.nextDouble() - 1;
         
        thetaOut = new double[numberOutputs];
        for (int i = 0; i < numberOutputs; i++)
            thetaOut[i] = 2*rand.nextDouble() - 1;

        // FIX: matrixMiddle must be 256 × numberMiddle
        matrixMiddle = new double[totalInputs][numberMiddle];
        for (int i = 0; i < totalInputs; i++)
            for (int j = 0; j < numberMiddle; j++)
                matrixMiddle[i][j] = 2*rand.nextDouble() - 1;

        matrixOut = new double[numberMiddle][numberOutputs];
        for (int i = 0; i < numberMiddle; i++)
            for (int j = 0; j < numberOutputs; j++)
                matrixOut[i][j] = 2*rand.nextDouble() - 1;
    }

    /*************************************************************************/
    
    //Method trains neural network
    public void train()
    {
        for (int i = 0; i < numberIterations; i++)
            for (int j = 0; j < numberRecords; j++)
            {
                forwardCalculation(records.get(j).attributes);
                
                backwardCalculation(records.get(j).className);
            }
    }

    /*************************************************************************/

    //Method performs forward pass - computes input/output
    private void forwardCalculation(int[][] trainingInput)
    {
        // FIX: flatten 16x16 → 256
        int index = 0;
        for (int i = 0; i < numberInputs; i++)
            for(int j = 0; j < numberInputs; j++)
                input[index++] = trainingInput[i][j];

        // hidden layer
        for (int i = 0; i < numberMiddle; i++)
        {
            double sum = 0;

            for (int j = 0; j < totalInputs; j++)
                sum += input[j] * matrixMiddle[j][i]; 
            
            sum += thetaMiddle[i];

            middle[i] = 1/(1 + Math.exp(-sum));
        }        

        // output layer
        for (int i = 0; i < numberOutputs; i++)
        {
            double sum = 0;

            for (int j = 0; j < numberMiddle; j++)
                sum += middle[j] * matrixOut[j][i]; 
            
            sum += thetaOut[i];

            output[i] = 1/(1 + Math.exp(-sum));
        }        
    }

    /*************************************************************************/

    //Method performs backward pass - computes errors, updates weights/thetas
    private void backwardCalculation(double[] trainingOutput)
    {
        for (int i = 0; i < numberOutputs; i++)
            errorOut[i] = output[i] * (1-output[i]) * (trainingOutput[i] - output[i]);

        for (int i = 0; i < numberMiddle; i++)
        {
            double sum = 0;

            for (int j = 0; j < numberOutputs; j++)
                sum += matrixOut[i][j] * errorOut[j];                
         
            errorMiddle[i] = middle[i] * (1-middle[i]) * sum;
        }

        // update hidden → output
        for (int i = 0; i < numberMiddle; i++)
            for (int j = 0; j < numberOutputs; j++)
                matrixOut[i][j] += rate * middle[i] * errorOut[j];

        // FIX: update ALL 256 input → hidden weights
        for (int i = 0; i < totalInputs; i++)
            for (int j = 0; j < numberMiddle; j++)
                matrixMiddle[i][j] += rate * input[i] * errorMiddle[j];

        for (int i = 0; i < numberOutputs; i++)
            thetaOut[i] += rate * errorOut[i];

        for (int i = 0; i < numberMiddle; i++)
            thetaMiddle[i] += rate * errorMiddle[i];
    }

    /*************************************************************************/
    
    //method tests an image and return a result 
    private double[] test(int[][] inputImage)
    {
        forwardCalculation(inputImage);
        
        return output;
    }
    
    /*************************************************************************/
    
    //Method validates the network using the data from a file
    
    public void validate(String validationFile) throws IOException
    {
         Scanner inFile = new Scanner(new File(validationFile));
         
         //read number of records
         int numberRecords = inFile.nextInt();
         
         //for each record
         for(int x=0; x<numberRecords; x++)
     	{
        	 //read actual input
         	int fileInput[][] = new int[numberInputs][numberInputs];
         	
         	for(int i=0; i<numberInputs ; i++)
         	{
         		for(int j=0; j<numberInputs ; j++)
         		{
         			fileInput[i][j] = inFile.nextInt();
         		}
         	}
     		
         	//read actual output
         	int actualOuput = inFile.nextInt();
     		
         	double actualName = 0;
         	
         	if(actualOuput == 0)
         		actualName =0;
         	else if(actualOuput == 1)
         		actualName = 0.5;
         	else if(actualOuput == 2)
         		actualName = 1.0;
         	
         	//find predicted output
         	double[] predictOutput = test(fileInput);

         	//count number of error
            for (int j = 0; j < numberOutputs; j++)
            {
           	 if(!checkMatch(decode(actualName),decode(predictOutput[j])))
           		 numberError += 1;
            }	

     	}
         
         //calculate the error rate
         errorRate = ((double)numberError/numberRecords) *100;
         
         //printing the validation error on the screen 
         System.out.println("Validation error : "+errorRate);
         
         inFile.close();
    }
    
    /*************************************************************************/

    //method converts output to a class based on nearest center
    public static String decode(double resultOutput)
    {
        // centers match your encoded class values
        double[] centers = {0.0, 0.5, 1.0};
        String[] classes = {"zero","one","two"};

        int best = 0;
        double bestDist = Math.abs(resultOutput - centers[0]);

        for (int i = 1; i < centers.length; i++)
        {
            double dist = Math.abs(resultOutput - centers[i]);
            if (dist < bestDist)
            {
                best = i;
                bestDist = dist;
            }
        }
        return classes[best];
    }
    
    
    /*************************************************************************/
    
    //method checks if predicted matches the actual 
    private boolean checkMatch(String predicted,String actual)
    {
    	if(predicted.equals(actual))
    		return true;
    	
    	return false;
    }
    
    /*************************************************************************/
    
    //Method reads inputs from input file, computes outputs, and writes outputs 
    //to output file
    public void testData(String inputFile, String outputFile) throws IOException
    {
    	Scanner inFile = new Scanner(new File(inputFile));
        PrintWriter outFile = new PrintWriter(new FileWriter(outputFile));
        
        //read number of records
        int numberRecords = inFile.nextInt();
        
        //for each record
        for(int x=0; x<numberRecords; x++)
    	{
       	 //read actual input
        	int fileInput[][] = new int[numberInputs][numberInputs];
        	
        	for(int i=0; i<numberInputs ; i++)
        	{
        		for(int j=0; j<numberInputs ; j++)
        		{
        			fileInput[i][j] = inFile.nextInt();
        		}
        	}
        	
        	//find predicted output
         	double[] predictOutput = test(fileInput);
         	
         	for(int y=0; y<predictOutput.length ; y++)
         	{
         		outFile.println(decode(predictOutput[y]));
         	}
         	
         	
    	}
        
        outFile.println();
        outFile.println("Error Rate : %" + errorRate);	  //print error rate
        outFile.println("Middle nodes : "+numberMiddle); //print middle nodes
        outFile.println("Learning rate : "+rate);		  //print learning rate
        outFile.println("Number iterations : "+numberIterations);//print number iterations
        outFile.println("Random number seed : "+seed);			  //print random number seed
        
        inFile.close();
        outFile.close();
        
    }
    
    
    
}
