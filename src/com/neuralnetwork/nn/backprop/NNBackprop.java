package com.neuralnetwork.nn.backprop;

import com.neuralnetwork.core.Example;
import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.interfaces.INeuralLayer;
import org.ejml.data.DenseMatrix64F;

/**
 * On a single training example,
 * a single iteration of the backprop algorithm
 */
class NNBackprop
{
    /**
     * This is the output of the current layer that's used as the input to the next layer
     * The last YInfo is actually the output of the network.
     *
     * To make array indexes easier, we don't actually store the input as a YInfo
     */
    protected class YInfo
    {
        /**
         * each value maps to a neuron
         * but first value is always = +1 (maps to bias)
         * so array length = numberOfNeurons + 1
         */
        public double[] yInducedLocalField;
        /**
         * each value maps to a neuron
         * but first value is always = +1 (maps to bias)
         * so array length = numberOfNeurons + 1
         */
        public double[] y;

        public YInfo(int numberOfNeuronsInLayer)
        {
            yInducedLocalField = new double[numberOfNeuronsInLayer + 1];
            y = new double[numberOfNeuronsInLayer + 1];

            yInducedLocalField[0] = 1;
            y[0] = 1;
        }
    }

    protected class GradientInfo
    {
        /**
         * each value maps to a neuron
         * and only neurons have these
         */
        public double[] gradients;

        public GradientInfo(int numberOfNeuronsInLayer)
        {
            gradients = new double[numberOfNeuronsInLayer];
        }
    }

    protected Example example;

    protected INeuralLayer[] aLayers;

    protected YInfo[] aYInfo;
    protected GradientInfo[] aGradientInfo;

    /**
     * These values haven't been multiplied by the learning term yet.
     * Each value maps to a weight.
     */
    protected DenseMatrix64F[] aCumulativeLearningTermsMinusEta;

    NNBackprop(INeuralLayer... aLayers)
    {
        this.aLayers = aLayers;

        aYInfo = new YInfo[aLayers.length];
        aGradientInfo = new GradientInfo[aLayers.length];
        aCumulativeLearningTermsMinusEta = new DenseMatrix64F[aLayers.length];

        for(int i=0; i<aLayers.length; i++)
        {
            int numberOfNeurons = aLayers[i].getNumberOfNeurons();

            aYInfo[i] = new YInfo(numberOfNeurons);
            aGradientInfo[i] = new GradientInfo(numberOfNeurons);

            DenseMatrix64F matrix = aLayers[i].getWeightMatrix();

            aCumulativeLearningTermsMinusEta[i] = new DenseMatrix64F(matrix.numRows, matrix.numCols);
        }
    }

    /**
     * On a single training example:
     *
     * 1. Perform the forward propagation. Store induced local fields and impulse function values.
     * 2. Perform the backpropagation. Store each neuron's gradient.
     * 3. Calculate the learning term for each weight of each neuron
     * 4. Return a matrix containing the learning terms
     *
     * WARNING: this is NOT thread-safe!!!
     *
     * @param example
     * @return the cumulative learning terms without eta
     */
    DenseMatrix64F[] go(Example example)
    {
        this.example = example;

        forwardProp();
        backprop();

        return updateCumulativeLearningTerms();
    }

    protected void forwardProp()
    {
        for(int i=0; i<aLayers.length; i++)
        {
            generateY(i, getY(i - 1));
        }
    }

    /**
     * Necessary because {@link #aYInfo} doesn't keep track of actual NN input.
     *
     * Need to call {@link #forwardProp()} in order to work.
     *
     * @param prevLayer >= -1 (prevLayer == -1 corresponds to input layer)

     * @return
     */
    protected double[] getY(int prevLayer)
    {
        if (prevLayer == -1)
        {
            return example.input;
        }

        return aYInfo[prevLayer].y;
    }

    protected void generateY(int index, double[] input)
    {
        double[] inducedLocalField = aLayers[index].generateInducedLocalField(input);

        System.arraycopy(inducedLocalField,0,
                aYInfo[index].yInducedLocalField,1,
                inducedLocalField.length);

        inducedLocalField = aYInfo[index].yInducedLocalField;

        IActivationFunction.IDifferentiableFunction phi = aLayers[index].getImpulseFunction();

        for(int i=1; i<inducedLocalField.length; i++) /** i=1 skips bias **/
            aYInfo[index].y[i] = phi.apply(inducedLocalField[i]);
    }

    protected void backprop()
    {
        for(int i=aLayers.length-1; i>=0 ;i--)
        {
            constructGradients(i);
        }
    }

    protected void constructGradients(int layer)
    {
        int numberOfNeurons = aLayers[layer].getNumberOfNeurons();

        GradientInfo gradientInfo = aGradientInfo[layer];

        for(int neuronIndex=0; neuronIndex<numberOfNeurons; neuronIndex++)
        {
            gradientInfo.gradients[neuronIndex] = gradient(layer, neuronIndex + 1);
        }
    }

    /**
     * Gradient for the given example, layer, and neuron
     *
     * @param layer
     * @param neuronNum must be >= 1. neuronIndex != nueronNum... indexes start at 0, neuronNum at 1
     * @return gradient value for the given layer, and neuron
     */
    protected double gradient(int layer, int neuronNum)
    {
        YInfo yInfo = aYInfo[layer];

        IActivationFunction.IDifferentiableFunction phi = aLayers[layer].getImpulseFunction();

        if (layer == aLayers.length-1)
        {
            // (oj - tj) * phi'_j(v^L_j)
            final double inducedLocalField = yInfo.yInducedLocalField[neuronNum];
            final double output = yInfo.y[neuronNum];

            return (example.expected[neuronNum] - output)
                    * phi.derivative(inducedLocalField);
        }
        else
        {
             return phi.derivative(yInfo.yInducedLocalField[neuronNum])
                     * sumGradients(neuronNum, layer + 1);
        }
    }

    /**
     * Find the sum of the gradients times the weights of the next layer
     * for the current neuron.
     *
     * The weights that connect neuron j are found in column (j+1) of the weight matrix
     * (the column with index j)
     *
     * @param j is the neuronNum where the neuronNum starts at 1.
     *          neuronIndex != neuronNum... indexes start at 0, neuronNum starts at 1
     * @param nextLayer <= total number of layers
     *
     * @return
     */
    protected double sumGradients(int j, int nextLayer)
    {
        double rslt = 0f;

        int numberOfNeurons = aLayers[nextLayer].getNumberOfNeurons();

        DenseMatrix64F weights = aLayers[nextLayer].getWeightMatrix();

        for(int neuronIndex=0; neuronIndex < numberOfNeurons; ++neuronIndex)
        {
            // w_ij^(l+1) * delta_i^(l+1)
            rslt += weights.unsafe_get(neuronIndex, j) * gradient(nextLayer, neuronIndex+1);
        }

        return rslt;
    }

    protected DenseMatrix64F[] updateCumulativeLearningTerms()
    {
        for(int layer=0; layer<aLayers.length; ++layer) {

            DenseMatrix64F learningMatrix = aCumulativeLearningTermsMinusEta[layer];

            for (int row = 0; row < learningMatrix.numRows; ++row)
                for (int col = 0; col < learningMatrix.numCols; ++col)
                {
                    double curTerm = learningMatrix.unsafe_get(row,col);

                    //Δw^l_kj = η δ^l_k * y^(l−1)_j

                    double gradient_k = aGradientInfo[layer].gradients[row]; //recall that rows in matrix correspond to neurons
                    double prevOutput_j = getY(layer - 1)[col];

                    learningMatrix.unsafe_set(row,col, curTerm + gradient_k * prevOutput_j);
                }
        }

        return aCumulativeLearningTermsMinusEta;
    }
}
