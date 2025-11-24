import java.io.IOException;
import java.util.Scanner;

public class Driver {

	public static void main(String[] args) throws IOException
	{
		//Scanner object 
		Scanner input = new Scanner(System.in); 
		
		//get the training data file 
		System.out.println("Enter the name of Training data file : ");
		String inputTrainingFile = input.nextLine(); 
		
		//get the validation data file 
		System.out.println("Enter the name of the validation file : ");
		String inputValidation = input.nextLine();
		
		//get the test data file
		System.out.println("Enter the name of Test data file : ");
		String inputTestFile = input.nextLine();

		//get the classified data file
		System.out.println("Enter the name of classified file :");
		String classifiedFile = input.nextLine();
		
		//create a neural netowrk object
		Neural neural = new Neural();
		
		//load training data into neural network model
		neural.loadTrainingFile(inputTrainingFile);
		
		//set parameters
		neural.setParameters(500, 1000000, 0.3, 1321998);
		
		//validate neural network
		neural.validate(inputValidation);
		
		//test data
		neural.testData(inputTestFile, classifiedFile);
		
		

	}

}
