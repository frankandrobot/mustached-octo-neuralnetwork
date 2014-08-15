package com.neuralnetwork;

import com.neuralnetwork.cnn.filter.SimpleConvolutionFilterTest;
import com.neuralnetwork.core.IActivationFunctionTest;
import com.neuralnetwork.nn.MultiLayerNNTest;
import com.neuralnetwork.nn.backprop.NNBackpropTest;
import com.neuralnetwork.nn.layer.NNLayerTest;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;

@RunWith(Suite.class)
@Suite.SuiteClasses({
        SimpleConvolutionFilterTest.class,
        IActivationFunctionTest.class,
        NNLayerTest.class,
        MultiLayerNNTest.class,
        NNBackpropTest.class
})
public class WorkingTests {

}
