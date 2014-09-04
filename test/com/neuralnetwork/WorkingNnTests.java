package com.neuralnetwork;

import com.neuralnetwork.nn.NNTest;
import com.neuralnetwork.nn.backprop.NNBackpropHelperTest;
import com.neuralnetwork.nn.backprop.NNBackpropTest;
import com.neuralnetwork.nn.backprop.XORBackPropagationTest;
import com.neuralnetwork.nn.backprop.XORNetworkTest;
import com.neuralnetwork.nn.layer.NNLayerTest;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;

@RunWith(Suite.class)
@Suite.SuiteClasses({
        NNLayerTest.class,
        NNTest.class,
        NNBackpropHelperTest.class,
        XORNetworkTest.class,
        XORBackPropagationTest.class,
        NNBackpropTest.class
})
public class WorkingNnTests
{

}
