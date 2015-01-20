package com.neuralnetwork.nn.layer;

import com.neuralnetwork.core.Dimension;
import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.interfaces.ICnnMap;
import com.neuralnetwork.core.interfaces.INnLayer;
import com.neuralnetwork.core.neuron.Neuron;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import java.util.List;

public class NNLayer implements INnLayer, ICnnMap
{
    /**
     * for simplicity we assume all neurons have the same activation function
     */
    protected IActivationFunction.IDifferentiableFunction phi;

    /**
     * For simplicity we assume that all neurons have the same number of weights.
     * (the network is fully connected)
     * So:
     *
     * number of neurons x number of weights in a single neuron
     *
     */
    protected DenseMatrix64F weights;

    /**
     * number of weights in a single nueron x 1
     */
    protected DenseMatrix64F mInput;
    /**
     * number of neurons x 1
     */
    protected DenseMatrix64F mOutput;
    /**
     * (number of neurons + 1) x 1
     * with mY[0] = +1 always
     */
    protected double[] y;

    protected int numberOfNuerons;
    protected int numberOfWeightsInSingleNueron;

    final protected Dimension inputDim;
    final protected Dimension outputDim;

    NNLayer(NNLayerBuilder builder)
    {
        Neuron[] neurons = builder.aNeurons.toArray(new Neuron[builder.aNeurons.size()]);

        numberOfNuerons = neurons.length;
        numberOfWeightsInSingleNueron = neurons[0].getNumberOfWeights();

        //get phi
        phi = neurons[0].phi();

        //build weight matrix = number of neurons x number of weights
        weights = new DenseMatrix64F(numberOfNuerons, numberOfWeightsInSingleNueron);

        for(int row = 0; row<weights.numRows; ++row)
            for(int col = 0; col<weights.numCols; ++col)
            {
                weights.unsafe_set(row,col, neurons[row].getWeight(col) );
            }

        //build input
        mInput = new DenseMatrix64F(numberOfWeightsInSingleNueron, 1);
        inputDim = new Dimension(mInput.numRows, mInput.numCols);

        //build output
        mOutput = new DenseMatrix64F(numberOfNuerons, 1);
        outputDim = new Dimension(mOutput.numRows, mOutput.numCols);

        //build y
        y = new double[numberOfNuerons + 1];
    }

    @Override
    public double[] generateY(double[] input)
    {
        double[] output = generateOutput(input);

        y[0] = 1;
        System.arraycopy(output,0, y,1, output.length);

        return y;
    }

    @Override
    public double[] generateOutput(double[] input)
    {
        double[] output = generateInducedLocalField(input);

        for(int i=0; i<output.length; ++i)
        {
            output[i] = phi.apply(output[i]);
        }

        return output;
    }

    @Override
    public double[] generateInducedLocalField(double[] input)
    {
        assert(input.length == getInputDim().rows);
        assert(input[0] == 1.0);

        //create column major matrix
        mInput.set(getInputDim().rows,1,false,input);
        CommonOps.mult(weights, mInput, mOutput);

        return mOutput.getData();
    }

    @Override
    public Dimension getInputDim()
    {
        return inputDim;
    }

    @Override
    public Dimension getOutputDim()
    {
        return outputDim;
    }

    @Override
    public int getNumberOfNeurons()
    {
        return numberOfNuerons;
    }

    @Override
    public DenseMatrix64F getWeightMatrix()
    {
        return weights;
    }

    @Override
    public IActivationFunction.IDifferentiableFunction getImpulseFunction()
    {
        return phi;
    }


    /**
     * When used as an {@link com.neuralnetwork.core.interfaces.ICnnMap}
     * each nueron is fully connected to every unit in the input
     *
     * assumes inputs are all same size
     *
     * @param inputs
     * @return
     */
    @Override
    public DenseMatrix64F generateOutput(DenseMatrix64F... inputs)
    {
        generateInducedLocalField(inputs);

        for(int i=0; i<mOutput.data.length; ++i)
        {
            mOutput.data[i] = phi.apply(mOutput.data[i]);
        }

        return mOutput;
    }

    @Override
    public DenseMatrix64F generateInducedLocalField(DenseMatrix64F... inputs)
    {
        //copy over input into mInput

        mInput.data[0] = 1.0;

        for(int i=0; i<inputs.length; ++i)
        {
            int length = inputs[i].data.length;

            System.arraycopy(inputs[i].data, 0, mInput.data, i*length+1, length);
        }


        //perform multiplication

        CommonOps.mult(weights, mInput, mOutput);

        return mOutput;
    }

    @Override
    public void validateInputs(List<ICnnMap> inputMaps)
    {
        int totalUnits = 0;

        for(ICnnMap map:inputMaps)
        {
            Dimension dims = map.getOutputDim();

            totalUnits += dims.cols * dims.rows;
        }

        int inputUnits = getInputDim().rows - 1;

        if (totalUnits != inputUnits)
            throw new IllegalArgumentException("Inputs don't match. "+totalUnits+"!="+inputUnits);
    }

    @Override
    public int getNumberOfInputs() {
        return 0;
    }
}
