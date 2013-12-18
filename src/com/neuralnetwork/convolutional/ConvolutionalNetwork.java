package com.neuralnetwork.convolutional;

import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.interfaces.INeuralNetwork;
import org.ejml.data.DenseMatrix64F;

import static com.neuralnetwork.core.interfaces.INeuralNetwork.IMatrixNeuralNetwork;

public class ConvolutionalNetwork
{
    protected LayorInfo[] aLayers;
    protected int numberLayers;

    protected ExampleInfo[] aExamples;
    protected int numberExamples;

    /**
     * Aka error.
     * An 1 x n matrix containing the differences squared of each training example.
     */
    protected DenseMatrix64F mTotalDifferenceSquared;

    final protected double alpha; /** momentum **/
    final protected double eta; /** learning parameter **/
    final protected int numberIterations;

    protected IActivationFunction.IDifferentiableFunction phi;

    final protected Output output = new Output();

    public ConvolutionalNetwork(Builder builder)
    {
        this.eta = builder.eta;
        this.alpha = builder.alpha;
        this.phi = builder.phi;
        this.numberIterations = builder.numberIterations == 0 ? 1000 : builder.numberIterations;

        initializeLayers(builder);
    }

    protected LayorInfo getLayer(int i)
    {
        return aLayers[i];
    }

    static public class Builder
    {
        private IActivationFunction.IDifferentiableFunction phi;
        protected IMatrixNeuralNetwork[] aLayers;
        protected Double alpha;
        protected Double eta;
        protected int numberIterations;

        public Builder setGlobalActivationFunction(IActivationFunction.IDifferentiableFunction phi)
        {
            this.phi = phi;
            return this;
        }

        public Builder setLayers(IMatrixNeuralNetwork... aLayers)
        {
            this.aLayers = aLayers;
            return this;
        }

        public Builder setMomentumParam(Double alpha)
        {
            this.alpha = alpha;
            return this;
        }

        public Builder setLearningParam(Double eta)
        {
            this.eta = eta;
            return this;
        }

        public Builder setIterations(int numberIterations)
        {
            this.numberIterations = numberIterations;
            return this;
        }
    }

    /**
     * (mExampleInput, mExpected) form the training example
     */
    protected class ExampleInfo
    {
        /**
         * These two form the actual training data pair
         */
        protected DenseMatrix64F mExampleInput;
        protected DenseMatrix64F mExpected;

        /**
         * Used to store the actual output for the given example input
         */
        protected DenseMatrix64F mActual;
        /**
         * If mActual and mExpected where vectors,
         * it stores:
         *  (mActual - mExpected).(mActual - mExpected)
         * i.e., element by element multiplication of (mActual - mExpected) with itself
         * followed by taking the sum of all the elements in the result.
         */
        protected double differenceSquared;
    }

    /**
     * This class wraps the neural network
     * with additional information.
     *
     * Basically it works per layer
     * and you get back info about its individual neurons
     */
    protected class LayorInfo
    {
        final public IMatrixNeuralNetwork layer;
        /**
         * For each neuron k in the layer,
         * store v_k (induced local field).
         */
        public DenseMatrix64F mInducedLocalField;
        /**
         * For each neuron k in the layer,
         * store value of impulse function.
         */
        public DenseMatrix64F mImpulseFunction;
        /**
         * Store the inputs (y_k) from the previous layer.
         */
        public DenseMatrix64F mInput;
        /**
         * For each neuron k in the layer,
         * store its gradient
         */
        public DenseMatrix64F mGradients;

        /**
         * For each neuron k in the layer,
         * aPrevWeights[k] is its previous set of weights.
         * aPrevWeights[k].get(j) is neuron k's jth previous weight.
         */
        public MNeuron[][] aPrevWeights;
        /**
         * For neuron k's jth weight in the layer,
         * aWeightAdjustments[k].get(j) is the quantity to add to its jth weight
         * when adjusting the weight with the backprop algorithm
         */
        public MNeuron[][] aWeightAdjustments;

        public LayorInfo(IMatrixNeuralNetwork layer)
        {
            this.layer = layer;
            //these are all created to be the size of the feature map
            final int n = (int)Math.sqrt(this.layer.getNumberOfNeurons());
            this.mInducedLocalField = new DenseMatrix64F(n,n);
            this.mImpulseFunction = new DenseMatrix64F(n,n);
            this.mGradients = new DenseMatrix64F(n,n);
            this.aPrevWeights = new MNeuron[n][n];
            this.aWeightAdjustments = new MNeuron[n][n];

            /*for(int i=0; i<n; i++)
                for(int j=0; j<n; j++)
                {
                    this.aPrevWeights[i][j] =
                this.aPrevWeights[len] = new NVector().setSize(neuron.getNumberOfWeights());
                this.aWeightAdjustments[len] = new NVector().setSize(neuron.getNumberOfWeights());
                ++len;
            }*/
        }
    }

    protected class DebugOutput
    {
        String weights()
        {
            String rslt = "";
            for(int neuron=0; neuron< aLayers[0].layer.getNumberOfNeurons(); neuron++)
            {
                rslt += String.format("%20s | %20s %n",
                        getNeuron(aLayers[0].layer, neuron),
                        getNeuron(aLayers[1] != null ? aLayers[1].layer : null, neuron));
            }
            return rslt;
        }

        private String getNeuron(INeuralNetwork network,
                                 int neuron)
        {
            if (network != null)
                return neuron < network.getNumberOfNeurons()
                        ? network.getNeuron(neuron).toString()
                        : "";
            return "";
        }

        /**
         * Dumps the debug info for the given backprop iteration
         *
         * @param iteration given iteration of the backprop algorithm
         */
        protected void backpropDump(int iteration)
        {
            System.out.println("====================================================================");
            System.out.println("Iteration = "+ iteration);
            System.out.format("Weights =%n%s", weights());
            //System.out.format("Error = %s %n", mTotalDifferenceSquared.sumOfCoords());
            System.out.format("%5s %20s %20s%n", "i", "Expected", "Actual");
            for(int i=0; i< numberExamples; ++i)
                System.out.format("%5s %20s %20s%n",
                        i,
                        aExamples[i].mExpected,
                        aExamples[i].mActual);
        }
    }

    protected void initializeLayers(Builder builder)
    {
        numberLayers = builder.aLayers.length;

        aLayers = new LayorInfo[numberLayers];

        for(int i=0; i<numberLayers; i++)
            aLayers[i] = new LayorInfo(builder.aLayers[i]);
    }

    public void setupExampleInfo(DenseMatrix64F... aInputExpected)
    {
        if (aInputExpected.length % 2 != 0)
            throw new IllegalArgumentException();

        numberExamples = aInputExpected.length / 2;

        aExamples = new ExampleInfo[numberExamples];

        //save aInputExpected
        for(int i=0; i< numberExamples; i++)
        {
            aExamples[i] = new ExampleInfo();
            aExamples[i].mExampleInput = aInputExpected[2*i];
            aExamples[i].mExpected = aInputExpected[2*i+1];
        }

        //setup error
        mTotalDifferenceSquared = new DenseMatrix64F(1, numberExamples);
    }

    /*public void backpropagation(double errorBound, NVector... aInputExpected)
    {
        if (aInputExpected.length % 2 != 0)
            throw new IllegalArgumentException();

        //initialization
        setupExampleInfo(aInputExpected);

        DebugOutput debugOutput = new DebugOutput();
        int iteration = 1;

        while( backpropagation() > errorBound && iteration <= numberIterations)
        {
            debugOutput.backpropDump(iteration);
            iteration++;
        }
        debugOutput.backpropDump(iteration);
    }*/

    public DenseMatrix64F output(DenseMatrix64F input)
    {
        if (aExamples == null)
        {
            setupExampleInfo(input, new DenseMatrix64F(1,1));
        }

        return output.output(0, 0, input);
    }

/*    *//**
     * Trains the network using pairs of inputs/expected values
     *
     * @return sum of the total differnce squared (error)
     *//*
    protected double backpropagation()
    {
        resetWeightAdjustments();

        for(int i=0; i< numberExamples; i++)
        {
            forwardPropagation(i);
            constructGradients(i);
            saveWeightAdjustments();
        }

        adjustWeights();

        for(int i=0; i< numberExamples; i++)
        {
            constructErrorFunction(i);
        }

        return mTotalDifferenceSquared.sumOfCoords();
    }

    protected void constructErrorFunction(int example)
    {
        NVector vActual = forwardPropagation(example);
        NVector vExpected = aExamples[example].mExpected;
        aExamples[example].differenceSquared = vExpected.subtract(vActual).dotProduct();
        mTotalDifferenceSquared.set(example, aExamples[example].differenceSquared);
    }*/

    /*protected NVector forwardPropagation(int example)
    {
        return output(example, 0, aExamples[example].mExampleInput);
    }*/

    protected class Output
    {
        /**
         * Find the actual output for the given example.
         *
         * Side effect: store stuff (induced local field, impulse function, etc) in the layer info
         *
         *
         * @param example
         * @param layer the example's layer
         * @return output
         */
        protected DenseMatrix64F output(int example, int layer, DenseMatrix64F input)
        {
            LayorInfo layorInfo = aLayers[layer];

            //save info for back propagation
            layorInfo.mInput = new DenseMatrix64F(input);

            constructInducedLocalField(layorInfo);
            constructImpulseFunction(layorInfo);

            if (layer < aLayers.length - 1)
            {
                return output(example, layer+1, aLayers[layer].mImpulseFunction);
            }
            else
            {
                aExamples[example].mActual = new DenseMatrix64F(aLayers[layer].mImpulseFunction);
            }
            return layorInfo.mImpulseFunction;
        }

        protected void constructInducedLocalField(LayorInfo layorInfo)
        {
            FeatureMap featureMap = ((FeatureMap)layorInfo.layer);
            //calculate v_k's
            for(int i=0; i<layorInfo.mInducedLocalField.numRows; i++)
                for(int j=0; j<layorInfo.mInducedLocalField.numCols; j++)
                {
                    double inducedLocalField = featureMap.rawoutput(layorInfo.mInput, i, j);
                    layorInfo.mInducedLocalField.unsafe_set(i, j, inducedLocalField);
                }
        }

        protected void constructImpulseFunction(LayorInfo layorInfo)
        {
            MNeuron sharedNeuron = layorInfo.layer.getNeuron(0);
            FeatureMap featureMap = ((FeatureMap)layorInfo.layer);
            //calculate y_k's aka output
            for(int i=0; i<layorInfo.mImpulseFunction.numRows; i++)
                for(int j=0; j<layorInfo.mImpulseFunction.numCols; j++)
                {
                    double inducedLocalField = layorInfo.mInducedLocalField.unsafe_get(i,j);
                    layorInfo.mImpulseFunction.unsafe_set(i, j, sharedNeuron.phi().apply(inducedLocalField));
                }
        }
    }
/*
    protected void constructGradients(int example)
    {
       constructGradients(example, aLayers.length - 1);
    }

    protected void constructGradients(int example, int layer)
    {
        if (layer >= 0)
        {
            LayorInfo layorInfo = aLayers[layer];

            for(int neuronPos=0; neuronPos<layorInfo.layer.getNumberOfNeurons(); neuronPos++)
            {
                layorInfo.vGradients.set(neuronPos, gradient(example, layer, neuronPos));
            }

            constructGradients(example, layer - 1);
        }
    }

    *//**
     * Gradient for the given example, layer, and neuron
     *
     * @param example example
     * @param layerLevel layer level
     * @param neuron neuron
     * @return gradient value for the given example, layer, and neuron
     *//*
    protected double gradient(int example, int layerLevel, int neuron)
    {
        if (layerLevel == aLayers.length-1)
        {
            // (oj - tj) * phi'_j(v^L_j)
            final double inducedLocalField = aLayers[layerLevel].vInducedLocalField.get(neuron);
            final double impulseFunction = aLayers[layerLevel].vImpulseFunction.get(neuron);
            return (aExamples[example].mExpected.get(neuron) - impulseFunction)
                    * phi.derivative(inducedLocalField);
        }
        else
        {
             return phi.derivative( aLayers[layerLevel].vInducedLocalField.get(neuron))
                     * sumGradients(example, layerLevel + 1, neuron);
        }
    }

    *//**
     * Find the sum of the gradients times the weights of the next layer
     * for the current neuron
     *
     *
     *
     * @param example
     * @param layerLevel layer level
     * @param currentNeuron current neuron
     * @return
     *//*
    protected double sumGradients(int example, int layerLevel, int currentNeuron)
    {
        LayorInfo layorInfo = aLayers[layerLevel];
        double rslt = 0f;

        int neuronPos = 0;
        for(Neuron neuron:layorInfo.layer)
        {
            rslt += neuron.getWeight(currentNeuron) * gradient(example, layerLevel, neuronPos++);
        }
        return rslt;
    }

    protected void saveWeightAdjustments()
    {
        saveWeightAdjustments(aLayers.length - 1);
    }

    protected void saveWeightAdjustments(int layer)
    {
        if (layer >= 0)
        {
            LayorInfo layorInfo = aLayers[layer];

            int neuronPos = 0;
            for(Neuron neuron:layorInfo.layer)
            {
                //iterate thru the neuron's weights
                for(int weight=0; weight<neuron.getNumberOfWeights(); weight++)
                {
                    double weightAdjustment =
                            eta * layorInfo.vGradients.get(neuronPos) * layorInfo.vInput.get(weight); //delta correction

                    //save weight adjustment
                    double curWeightAdjustment = layorInfo.aWeightAdjustments[neuronPos].get(weight);
                    layorInfo.aWeightAdjustments[neuronPos].set(weight, curWeightAdjustment + weightAdjustment);
                }

                neuronPos++;
            }

            saveWeightAdjustments(layer - 1);
        }
    }

    protected void adjustWeights()
    {
        for(LayorInfo layorInfo:aLayers)
        {
            //iterate thru the neurons in the layer
            int neuronPos = 0;
            for(Neuron neuron:layorInfo.layer)
            {
                //iterate thru the neuron's weights
                for(int weight=0; weight<neuron.getNumberOfWeights(); weight++)
                {
                    //adjust weights
                    double momentum = alpha * layorInfo.aPrevWeights[neuronPos].get(weight);
                    double curWeight = neuron.getWeight(weight);

                    //backup previous weight
                    layorInfo.aPrevWeights[neuronPos].set(weight, neuron.getWeight(weight));

                    neuron.setWeight(weight,
                                     curWeight
                                     + momentum
                                     + layorInfo.aWeightAdjustments[neuronPos].get(weight) // total delta weights
                    );
                }
                neuronPos++;
            }
        }
    }

    protected void resetWeightAdjustments()
    {
        for(LayorInfo layorInfo:aLayers)
        {
            //iterate thru the neurons in the layer
            int neuronPos = 0;
            for(Neuron neuron:layorInfo.layer)
            {
                //iterate thru the neuron's weights
                for(int weight=0; weight<neuron.getNumberOfWeights(); weight++)
                {
                    layorInfo.aWeightAdjustments[neuronPos].set(weight, 0);
                }
                neuronPos++;
            }
        }
    }

    public NVector rawoutput(NVector input)
    {
        throw new NotImplementedException();
    }*/

}
