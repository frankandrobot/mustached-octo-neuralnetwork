package com.neuralnetwork.core.backprop;

import com.neuralnetwork.core.Example;
import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.interfaces.INeuralLayer;
import org.ejml.data.DenseMatrix64F;

/**
 * On a single training example,
 * A single iteration of the backprop algorithm
 * on a single example
 */
class NNBackprop
{
    protected class OutputInfo
    {
        /**
         * each neuron has one
         */
        public double[] inducedLocalField;
        /**
         * each neuron has one
         */
        public double[] output;

        public OutputInfo(int numberOfNeuronsInLayer)
        {
            inducedLocalField = new double[numberOfNeuronsInLayer];
            output = new double[numberOfNeuronsInLayer];
        }
    }

    protected class GradientInfo
    {
        /**
         * each neuron has one
         */
        public double[] gradients;

        public GradientInfo(int numberOfNeuronsInLayer)
        {
            gradients = new double[numberOfNeuronsInLayer];
        }
    }

    protected Example example;

    protected INeuralLayer[] aLayers;

    protected OutputInfo[] aOutputInfo;
    protected GradientInfo[] aGradientInfo;

    NNBackprop(INeuralLayer... aLayers)
    {
        this.aLayers = aLayers;

        aOutputInfo = new OutputInfo[aLayers.length];
        aGradientInfo = new GradientInfo[aLayers.length];

        for(int i=0; i<aLayers.length; i++)
        {
            int size = aLayers[i].getNumberOfNeurons();

            aOutputInfo[i] = new OutputInfo(size);
            aGradientInfo[i] = new GradientInfo(size);
        }
    }

    /**
     * On a single training example,
     * 1. Perform the forward propagation. Store induced local fields and impulse function values.
     * 2. Perform the backpropagation. Store each neuron's gradient.
     * 3. Calculate the learning term for each weight of each neuron
     * 4. Return a matrix containing the learning terms
     *
     * @param example
     * @return
     */
    NNBackprop go(Example example)
    {
        this.example = example;

        forwardProp();

        return this;
    }

    protected void forwardProp()
    {
        setOutput(0, example.input);

        for(int i=1; i<aLayers.length; i++)
        {
            setOutput(i, aOutputInfo[i - 1].output);
        }
    }

    protected void setOutput(int index, double[] input)
    {
        aOutputInfo[index].inducedLocalField = aLayers[index].generateInducedLocalField(input);
        aOutputInfo[index].output = aLayers[index].generateOutput(input);
    }

    /**
     * Requires that you call forwardProp first
     *
     * @return
     */
    protected void backprop()
    {
        for(int i=aLayers.length-1; i>=0 ;i--)
        {
            constructGradients(i);
        }
    }

    protected void constructGradients(int layerIndex)
    {
        INeuralLayer layer = aLayers[layerIndex];

        DenseMatrix64F weightMatrix = layer.getWeightMatrix();

        for(int neuronPos=0; neuronPos<layer.getNumberOfNeurons(); neuronPos++)
        {
            aGradientInfo[layerIndex].gradients[neuronPos] = gradient(layerIndex, neuronPos);
        }
    }

    /**
     * Gradient for the given example, layer, and neuron
     *
     * @param layer
     * @param neuron
     * @return gradient value for the given layer, and neuron
     */
    protected double gradient(int layer, int neuron)
    {
        OutputInfo outputInfo = aOutputInfo[layer];
        IActivationFunction.IDifferentiableFunction phi = aLayers[layer].getImpulseFunction();

        if (layer == aLayers.length-1)
        {
            // (oj - tj) * phi'_j(v^L_j)
            final double inducedLocalField = outputInfo.inducedLocalField[neuron];
            final double output = outputInfo.output[neuron];

            return (example.expected[neuron] - output)
                    * phi.derivative(inducedLocalField);
        }
        else
        {
             return phi.derivative(outputInfo.inducedLocalField[neuron])
                     * sumGradients(neuron, layer + 1);
        }
    }

    /**
     * Find the sum of the gradients times the weights of the next layer
     * for the current neuron.
     *
     * The weights that connect neuron j are found in column j of the weight matrix.
     *
     * @param j neuron j
     * @param nextLayer
     *
     * @return
     */
    protected double sumGradients(int j, int nextLayer)
    {
        double rslt = 0f;
        DenseMatrix64F matrix = aLayers[nextLayer].getWeightMatrix();

        for(int neuron=0; neuron < matrix.numRows; ++neuron)
        {
            // w_ij^(l+1) * delta_i^(l+1)
            rslt += matrix.unsafe_get(neuron, j) * gradient(nextLayer, neuron);
        }

        return rslt;
    }
}
