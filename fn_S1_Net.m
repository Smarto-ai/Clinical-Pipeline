%%%% [S1-Net] For EBT
function lgraph = fn_S1_Net() 

lgraph = layerGraph();

tempLayers = imageInputLayer([224 224 3],"Name","input_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([7 7],64,"Name","conv1","Padding",[3 3 3 3],"Stride",[2 2])
    batchNormalizationLayer("Name","bn_conv1","Epsilon",0.001)
    reluLayer("Name","activation_1_relu")
    maxPooling2dLayer([3 3],"Name","max_pooling2d_1","Padding",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2a_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2a_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_2_relu")
    convolution2dLayer([3 3],64,"Name","res2a_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn2a_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_3_relu")
    convolution2dLayer([1 1],256,"Name","res2a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2a_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res2a_branch1","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2a_branch1","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_1")
    reluLayer("Name","activation_4_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2b_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2b_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_5_relu")
    convolution2dLayer([3 3],64,"Name","res2b_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn2b_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_6_relu")
    convolution2dLayer([1 1],256,"Name","res2b_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2b_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_2")
    reluLayer("Name","activation_7_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2c_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2c_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_8_relu")
    convolution2dLayer([3 3],64,"Name","res2c_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn2c_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_9_relu")
    convolution2dLayer([1 1],256,"Name","res2c_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2c_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_3")
    reluLayer("Name","activation_10_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3a_branch2a","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_11_relu")
    convolution2dLayer([3 3],128,"Name","res3a_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn3a_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_12_relu")
    convolution2dLayer([1 1],512,"Name","res3a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3a_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res3a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch1","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_4")
    reluLayer("Name","activation_13_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3b_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3b_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_14_relu")
    convolution2dLayer([3 3],128,"Name","res3b_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn3b_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_15_relu")
    convolution2dLayer([1 1],512,"Name","res3b_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3b_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_5")
    reluLayer("Name","activation_16_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3c_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3c_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_17_relu")
    convolution2dLayer([3 3],128,"Name","res3c_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn3c_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_18_relu")
    convolution2dLayer([1 1],512,"Name","res3c_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3c_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_6")
    reluLayer("Name","activation_19_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3d_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3d_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_20_relu")
    convolution2dLayer([3 3],128,"Name","res3d_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn3d_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_21_relu")
    convolution2dLayer([1 1],512,"Name","res3d_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3d_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_7")
    reluLayer("Name","activation_22_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4a_branch2a","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn4a_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_23_relu")
    convolution2dLayer([3 3],256,"Name","res4a_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4a_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_24_relu")
    convolution2dLayer([1 1],1024,"Name","res4a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4a_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],1024,"Name","res4a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn4a_branch1","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_8")
    reluLayer("Name","activation_25_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4b_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_26_relu")
    convolution2dLayer([3 3],256,"Name","res4b_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4b_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_27_relu")
    convolution2dLayer([1 1],1024,"Name","res4b_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_9")
    reluLayer("Name","activation_28_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4c_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4c_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_29_relu")
    convolution2dLayer([3 3],256,"Name","res4c_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4c_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_30_relu")
    convolution2dLayer([1 1],1024,"Name","res4c_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4c_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_10")
    reluLayer("Name","activation_31_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4d_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4d_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_32_relu")
    convolution2dLayer([3 3],256,"Name","res4d_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4d_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_33_relu")
    convolution2dLayer([1 1],1024,"Name","res4d_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4d_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_11")
    reluLayer("Name","activation_34_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4e_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4e_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_35_relu")
    convolution2dLayer([3 3],256,"Name","res4e_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4e_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_36_relu")
    convolution2dLayer([1 1],1024,"Name","res4e_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4e_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_12")
    reluLayer("Name","activation_37_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4f_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4f_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_38_relu")
    convolution2dLayer([3 3],256,"Name","res4f_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4f_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_39_relu")
    convolution2dLayer([1 1],1024,"Name","res4f_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4f_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_13")
    reluLayer("Name","activation_40_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5a_branch2a","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn5a_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_41_relu")
    convolution2dLayer([3 3],512,"Name","res5a_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn5a_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_42_relu")
    convolution2dLayer([1 1],2048,"Name","res5a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5a_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],2048,"Name","res5a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn5a_branch1","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_14")
    reluLayer("Name","activation_43_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5b_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5b_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_44_relu")
    convolution2dLayer([3 3],512,"Name","res5b_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn5b_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_45_relu")
    convolution2dLayer([1 1],2048,"Name","res5b_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5b_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_15")
    reluLayer("Name","activation_46_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5c_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5c_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_47_relu")
    convolution2dLayer([3 3],512,"Name","res5c_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn5c_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_48_relu")
    convolution2dLayer([1 1],2048,"Name","res5c_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5c_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_16")
    reluLayer("Name","activation_49_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","e1_conv_1")
    batchNormalizationLayer("Name","e1_bn_1")
    reluLayer("Name","e1_relu_1")
    maxPooling2dLayer([3 3],"Name","e1_pool_1","Padding",[2 2 2 2],"Stride",[2 2])
    sigmoidLayer("Name","e1_weight_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","e1_mul_1")
    convolution2dLayer([3 3],1024,"Name","e2_conv_1")
    batchNormalizationLayer("Name","e2_bn_1")
    reluLayer("Name","e2_relu_1")
    maxPooling2dLayer([3 3],"Name","e2_pool_1","Padding",[2 2 2 2],"Stride",[2 2])
    sigmoidLayer("Name","e2_weight_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","e1_mul_2")
    convolution2dLayer([3 3],2048,"Name","e3_conv_1")
    batchNormalizationLayer("Name","e3_bn_1")
    reluLayer("Name","e3_relu_1")
    maxPooling2dLayer([3 3],"Name","e3_pool_1","Padding",[2 2 2 2],"Stride",[2 2])
    sigmoidLayer("Name","e3_weight_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","e1_mul_3")
    convolution2dLayer([1 1],2048,"Name","e4_conv_1")
    batchNormalizationLayer("Name","e4_bn_1")
    reluLayer("Name","e4_relu_1")
    sigmoidLayer("Name","e4_weight_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","e5_conv_1","Padding","same")
    batchNormalizationLayer("Name","e5_bn_1")
    reluLayer("Name","e5_relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","e5_conv_2","DilationFactor",[2 2],"Padding","same")
    batchNormalizationLayer("Name","e5_bn_2")
    reluLayer("Name","e5_relu_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","e5_conv_3","DilationFactor",[4 4],"Padding","same")
    batchNormalizationLayer("Name","e5_bn_3")
    reluLayer("Name","e5_relu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","e5_conv_4","DilationFactor",[6 6],"Padding","same")
    batchNormalizationLayer("Name","e5_bn_4")
    reluLayer("Name","e5_relu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","e5_cat")
    convolution2dLayer([1 1],2048,"Name","e5_conv_5")
    reluLayer("Name","e5_relu_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],96,"Name","dec1_conv_1")
    batchNormalizationLayer("Name","dec1_bn_1")
    reluLayer("Name","dec1_relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","c_mul_1")
    convolution2dLayer([1 1],128,"Name","dec1_conv_2")
    batchNormalizationLayer("Name","dec1_bn_2")
    reluLayer("Name","dec1_relu_2")
    transposedConv2dLayer([2 2],128,"Name","dec1_tconv_1","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = crop2dLayer("centercrop","Name","dec1_crop_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","dec1_cat_1")
    convolution2dLayer([1 1],128,"Name","dec2_conv_2")
    batchNormalizationLayer("Name","dec2_bn_2")
    reluLayer("Name","dec2_relu_2")
    transposedConv2dLayer([2 2],128,"Name","dec2_tconv_1","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],96,"Name","dec2_conv_1")
    batchNormalizationLayer("Name","dec2_bn_1")
    reluLayer("Name","dec2_relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = crop2dLayer("centercrop","Name","dec2_crop_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","dec2_cat_1")
    convolution2dLayer([1 1],128,"Name","dec3_conv_2","Padding","same")
    sigmoidLayer("Name","dec3_weight_2")
    batchNormalizationLayer("Name","dec3_bn_2")
    reluLayer("Name","dec3_relu_2")
    transposedConv2dLayer([2 2],128,"Name","dec3_tconv_1","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],96,"Name","dec3_conv_1")
    batchNormalizationLayer("Name","dec3_bn_1")
    reluLayer("Name","dec3_relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = crop2dLayer("centercrop","Name","dec3_crop_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","dec3_cat_1")
    convolution2dLayer([3 3],128,"Name","dec_conv_1","Padding","same")
    batchNormalizationLayer("Name","dec_bn_1")
    reluLayer("Name","dec_relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","d1_conv_1","Padding","same")
    batchNormalizationLayer("Name","d1_bn_1")
    reluLayer("Name","d1_relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","d1_conv_2","DilationFactor",[2 2],"Padding","same")
    batchNormalizationLayer("Name","d1_bn_2")
    reluLayer("Name","d1_relu_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","d1_conv_3","DilationFactor",[4 4],"Padding","same")
    batchNormalizationLayer("Name","d1_bn_3")
    reluLayer("Name","d1_relu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","d1_conv_4","DilationFactor",[8 8],"Padding","same")
    batchNormalizationLayer("Name","d1_bn_4")
    reluLayer("Name","d1_relu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","d1_cat")
    convolution2dLayer([1 1],2,"Name","dec_scorer_1")
    reluLayer("Name","d1_relu_5")
    transposedConv2dLayer([4 4],2,"Name","dec_tconv_1","Stride",[4 4])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    crop2dLayer("centercrop","Name","dec_crop_1")
    softmaxLayer("Name","softmax-out")
    pixelClassificationLayer("Name","pixel_classification")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"input_1","conv1");
lgraph = connectLayers(lgraph,"input_1","dec_crop_1/ref");
lgraph = connectLayers(lgraph,"max_pooling2d_1","res2a_branch2a");
lgraph = connectLayers(lgraph,"max_pooling2d_1","res2a_branch1");
lgraph = connectLayers(lgraph,"max_pooling2d_1","e1_conv_1");
lgraph = connectLayers(lgraph,"bn2a_branch2c","add_1/in1");
lgraph = connectLayers(lgraph,"bn2a_branch1","add_1/in2");
lgraph = connectLayers(lgraph,"activation_4_relu","res2b_branch2a");
lgraph = connectLayers(lgraph,"activation_4_relu","add_2/in2");
lgraph = connectLayers(lgraph,"bn2b_branch2c","add_2/in1");
lgraph = connectLayers(lgraph,"activation_7_relu","res2c_branch2a");
lgraph = connectLayers(lgraph,"activation_7_relu","add_3/in2");
lgraph = connectLayers(lgraph,"bn2c_branch2c","add_3/in1");
lgraph = connectLayers(lgraph,"activation_10_relu","res3a_branch2a");
lgraph = connectLayers(lgraph,"activation_10_relu","res3a_branch1");
lgraph = connectLayers(lgraph,"activation_10_relu","dec3_conv_1");
lgraph = connectLayers(lgraph,"bn3a_branch2c","add_4/in1");
lgraph = connectLayers(lgraph,"bn3a_branch1","add_4/in2");
lgraph = connectLayers(lgraph,"activation_13_relu","res3b_branch2a");
lgraph = connectLayers(lgraph,"activation_13_relu","add_5/in2");
lgraph = connectLayers(lgraph,"activation_13_relu","e1_mul_1/in2");
lgraph = connectLayers(lgraph,"bn3b_branch2c","add_5/in1");
lgraph = connectLayers(lgraph,"activation_16_relu","res3c_branch2a");
lgraph = connectLayers(lgraph,"activation_16_relu","add_6/in2");
lgraph = connectLayers(lgraph,"bn3c_branch2c","add_6/in1");
lgraph = connectLayers(lgraph,"activation_19_relu","res3d_branch2a");
lgraph = connectLayers(lgraph,"activation_19_relu","add_7/in2");
lgraph = connectLayers(lgraph,"bn3d_branch2c","add_7/in1");
lgraph = connectLayers(lgraph,"activation_22_relu","res4a_branch2a");
lgraph = connectLayers(lgraph,"activation_22_relu","res4a_branch1");
lgraph = connectLayers(lgraph,"activation_22_relu","dec2_conv_1");
lgraph = connectLayers(lgraph,"bn4a_branch2c","add_8/in1");
lgraph = connectLayers(lgraph,"bn4a_branch1","add_8/in2");
lgraph = connectLayers(lgraph,"activation_25_relu","res4b_branch2a");
lgraph = connectLayers(lgraph,"activation_25_relu","add_9/in2");
lgraph = connectLayers(lgraph,"activation_25_relu","e1_mul_2/in2");
lgraph = connectLayers(lgraph,"bn4b_branch2c","add_9/in1");
lgraph = connectLayers(lgraph,"activation_28_relu","res4c_branch2a");
lgraph = connectLayers(lgraph,"activation_28_relu","add_10/in2");
lgraph = connectLayers(lgraph,"bn4c_branch2c","add_10/in1");
lgraph = connectLayers(lgraph,"activation_31_relu","res4d_branch2a");
lgraph = connectLayers(lgraph,"activation_31_relu","add_11/in2");
lgraph = connectLayers(lgraph,"bn4d_branch2c","add_11/in1");
lgraph = connectLayers(lgraph,"activation_34_relu","res4e_branch2a");
lgraph = connectLayers(lgraph,"activation_34_relu","add_12/in2");
lgraph = connectLayers(lgraph,"bn4e_branch2c","add_12/in1");
lgraph = connectLayers(lgraph,"activation_37_relu","res4f_branch2a");
lgraph = connectLayers(lgraph,"activation_37_relu","add_13/in2");
lgraph = connectLayers(lgraph,"bn4f_branch2c","add_13/in1");
lgraph = connectLayers(lgraph,"activation_40_relu","res5a_branch2a");
lgraph = connectLayers(lgraph,"activation_40_relu","res5a_branch1");
lgraph = connectLayers(lgraph,"activation_40_relu","dec1_conv_1");
lgraph = connectLayers(lgraph,"bn5a_branch2c","add_14/in1");
lgraph = connectLayers(lgraph,"bn5a_branch1","add_14/in2");
lgraph = connectLayers(lgraph,"activation_43_relu","res5b_branch2a");
lgraph = connectLayers(lgraph,"activation_43_relu","add_15/in2");
lgraph = connectLayers(lgraph,"activation_43_relu","e1_mul_3/in2");
lgraph = connectLayers(lgraph,"bn5b_branch2c","add_15/in1");
lgraph = connectLayers(lgraph,"activation_46_relu","res5c_branch2a");
lgraph = connectLayers(lgraph,"activation_46_relu","add_16/in2");
lgraph = connectLayers(lgraph,"bn5c_branch2c","add_16/in1");
lgraph = connectLayers(lgraph,"activation_49_relu","e5_conv_1");
lgraph = connectLayers(lgraph,"activation_49_relu","e5_conv_2");
lgraph = connectLayers(lgraph,"activation_49_relu","e5_conv_3");
lgraph = connectLayers(lgraph,"activation_49_relu","e5_conv_4");
lgraph = connectLayers(lgraph,"e1_weight_1","e1_mul_1/in1");
lgraph = connectLayers(lgraph,"e2_weight_1","e1_mul_2/in1");
lgraph = connectLayers(lgraph,"e3_weight_1","e1_mul_3/in1");
lgraph = connectLayers(lgraph,"e4_weight_1","c_mul_1/in2");
lgraph = connectLayers(lgraph,"e5_relu_1","e5_cat/in1");
lgraph = connectLayers(lgraph,"e5_relu_2","e5_cat/in2");
lgraph = connectLayers(lgraph,"e5_relu_3","e5_cat/in3");
lgraph = connectLayers(lgraph,"e5_relu_4","e5_cat/in4");
lgraph = connectLayers(lgraph,"e5_relu_5","c_mul_1/in1");
lgraph = connectLayers(lgraph,"dec1_relu_1","dec1_crop_1/ref");
lgraph = connectLayers(lgraph,"dec1_relu_1","dec1_cat_1/in1");
lgraph = connectLayers(lgraph,"dec1_tconv_1","dec1_crop_1/in");
lgraph = connectLayers(lgraph,"dec1_crop_1","dec1_cat_1/in2");
lgraph = connectLayers(lgraph,"dec2_tconv_1","dec2_crop_1/in");
lgraph = connectLayers(lgraph,"dec2_relu_1","dec2_crop_1/ref");
lgraph = connectLayers(lgraph,"dec2_relu_1","dec2_cat_1/in1");
lgraph = connectLayers(lgraph,"dec2_crop_1","dec2_cat_1/in2");
lgraph = connectLayers(lgraph,"dec3_tconv_1","dec3_crop_1/in");
lgraph = connectLayers(lgraph,"dec3_relu_1","dec3_crop_1/ref");
lgraph = connectLayers(lgraph,"dec3_relu_1","dec3_cat_1/in1");
lgraph = connectLayers(lgraph,"dec3_crop_1","dec3_cat_1/in2");
lgraph = connectLayers(lgraph,"dec_relu_1","d1_conv_1");
lgraph = connectLayers(lgraph,"dec_relu_1","d1_conv_2");
lgraph = connectLayers(lgraph,"dec_relu_1","d1_conv_3");
lgraph = connectLayers(lgraph,"dec_relu_1","d1_conv_4");
lgraph = connectLayers(lgraph,"d1_relu_1","d1_cat/in1");
lgraph = connectLayers(lgraph,"d1_relu_2","d1_cat/in2");
lgraph = connectLayers(lgraph,"d1_relu_3","d1_cat/in3");
lgraph = connectLayers(lgraph,"d1_relu_4","d1_cat/in4");
lgraph = connectLayers(lgraph,"dec_tconv_1","dec_crop_1/in");

end

