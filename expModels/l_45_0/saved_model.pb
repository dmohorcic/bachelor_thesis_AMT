??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12unknown8??
?
conv2d_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_nameconv2d_40/kernel
}
$conv2d_40/kernel/Read/ReadVariableOpReadVariableOpconv2d_40/kernel*&
_output_shapes
:(*
dtype0
t
conv2d_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameconv2d_40/bias
m
"conv2d_40/bias/Read/ReadVariableOpReadVariableOpconv2d_40/bias*
_output_shapes
:(*
dtype0
?
conv2d_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:(`*!
shared_nameconv2d_41/kernel
}
$conv2d_41/kernel/Read/ReadVariableOpReadVariableOpconv2d_41/kernel*&
_output_shapes
:(`*
dtype0
t
conv2d_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_nameconv2d_41/bias
m
"conv2d_41/bias/Read/ReadVariableOpReadVariableOpconv2d_41/bias*
_output_shapes
:`*
dtype0
|
dense_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_45/kernel
u
#dense_45/kernel/Read/ReadVariableOpReadVariableOpdense_45/kernel* 
_output_shapes
:
??*
dtype0
s
dense_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_45/bias
l
!dense_45/bias/Read/ReadVariableOpReadVariableOpdense_45/bias*
_output_shapes	
:?*
dtype0
{
dense_46/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?X* 
shared_namedense_46/kernel
t
#dense_46/kernel/Read/ReadVariableOpReadVariableOpdense_46/kernel*
_output_shapes
:	?X*
dtype0
r
dense_46/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:X*
shared_namedense_46/bias
k
!dense_46/bias/Read/ReadVariableOpReadVariableOpdense_46/bias*
_output_shapes
:X*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
v
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0
y
true_positives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nametrue_positives_2
r
$true_positives_2/Read/ReadVariableOpReadVariableOptrue_positives_2*
_output_shapes	
:?*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:?*
dtype0
{
false_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namefalse_positives_1
t
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes	
:?*
dtype0
{
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namefalse_negatives_1
t
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_40/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameAdam/conv2d_40/kernel/m
?
+Adam/conv2d_40/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/kernel/m*&
_output_shapes
:(*
dtype0
?
Adam/conv2d_40/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*&
shared_nameAdam/conv2d_40/bias/m
{
)Adam/conv2d_40/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/bias/m*
_output_shapes
:(*
dtype0
?
Adam/conv2d_41/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(`*(
shared_nameAdam/conv2d_41/kernel/m
?
+Adam/conv2d_41/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/kernel/m*&
_output_shapes
:(`*
dtype0
?
Adam/conv2d_41/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*&
shared_nameAdam/conv2d_41/bias/m
{
)Adam/conv2d_41/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/bias/m*
_output_shapes
:`*
dtype0
?
Adam/dense_45/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_45/kernel/m
?
*Adam/dense_45/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_45/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_45/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_45/bias/m
z
(Adam/dense_45/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_45/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_46/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?X*'
shared_nameAdam/dense_46/kernel/m
?
*Adam/dense_46/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_46/kernel/m*
_output_shapes
:	?X*
dtype0
?
Adam/dense_46/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:X*%
shared_nameAdam/dense_46/bias/m
y
(Adam/dense_46/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_46/bias/m*
_output_shapes
:X*
dtype0
?
Adam/conv2d_40/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_nameAdam/conv2d_40/kernel/v
?
+Adam/conv2d_40/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/kernel/v*&
_output_shapes
:(*
dtype0
?
Adam/conv2d_40/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*&
shared_nameAdam/conv2d_40/bias/v
{
)Adam/conv2d_40/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/bias/v*
_output_shapes
:(*
dtype0
?
Adam/conv2d_41/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(`*(
shared_nameAdam/conv2d_41/kernel/v
?
+Adam/conv2d_41/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/kernel/v*&
_output_shapes
:(`*
dtype0
?
Adam/conv2d_41/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*&
shared_nameAdam/conv2d_41/bias/v
{
)Adam/conv2d_41/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/bias/v*
_output_shapes
:`*
dtype0
?
Adam/dense_45/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_45/kernel/v
?
*Adam/dense_45/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_45/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_45/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_45/bias/v
z
(Adam/dense_45/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_45/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_46/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?X*'
shared_nameAdam/dense_46/kernel/v
?
*Adam/dense_46/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_46/kernel/v*
_output_shapes
:	?X*
dtype0
?
Adam/dense_46/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:X*%
shared_nameAdam/dense_46/bias/v
y
(Adam/dense_46/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_46/bias/v*
_output_shapes
:X*
dtype0

NoOpNoOp
?C
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?C
value?CB?C B?C
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
 bias
!trainable_variables
"	variables
#regularization_losses
$	keras_api
R
%trainable_variables
&	variables
'regularization_losses
(	keras_api
R
)trainable_variables
*	variables
+regularization_losses
,	keras_api
R
-trainable_variables
.	variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
R
7trainable_variables
8	variables
9regularization_losses
:	keras_api
h

;kernel
<bias
=trainable_variables
>	variables
?regularization_losses
@	keras_api
?
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_ratem?m?m? m?1m?2m?;m?<m?v?v?v? v?1v?2v?;v?<v?
8
0
1
2
 3
14
25
;6
<7
8
0
1
2
 3
14
25
;6
<7
 
?
Flayer_metrics
Gmetrics
trainable_variables
	variables

Hlayers
Inon_trainable_variables
regularization_losses
Jlayer_regularization_losses
 
\Z
VARIABLE_VALUEconv2d_40/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_40/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Klayer_metrics
Lmetrics
trainable_variables
	variables

Mlayers
Nnon_trainable_variables
regularization_losses
Olayer_regularization_losses
 
 
 
?
Player_metrics
Qmetrics
trainable_variables
	variables

Rlayers
Snon_trainable_variables
regularization_losses
Tlayer_regularization_losses
 
 
 
?
Ulayer_metrics
Vmetrics
trainable_variables
	variables

Wlayers
Xnon_trainable_variables
regularization_losses
Ylayer_regularization_losses
\Z
VARIABLE_VALUEconv2d_41/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_41/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
?
Zlayer_metrics
[metrics
!trainable_variables
"	variables

\layers
]non_trainable_variables
#regularization_losses
^layer_regularization_losses
 
 
 
?
_layer_metrics
`metrics
%trainable_variables
&	variables

alayers
bnon_trainable_variables
'regularization_losses
clayer_regularization_losses
 
 
 
?
dlayer_metrics
emetrics
)trainable_variables
*	variables

flayers
gnon_trainable_variables
+regularization_losses
hlayer_regularization_losses
 
 
 
?
ilayer_metrics
jmetrics
-trainable_variables
.	variables

klayers
lnon_trainable_variables
/regularization_losses
mlayer_regularization_losses
[Y
VARIABLE_VALUEdense_45/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_45/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
?
nlayer_metrics
ometrics
3trainable_variables
4	variables

players
qnon_trainable_variables
5regularization_losses
rlayer_regularization_losses
 
 
 
?
slayer_metrics
tmetrics
7trainable_variables
8	variables

ulayers
vnon_trainable_variables
9regularization_losses
wlayer_regularization_losses
[Y
VARIABLE_VALUEdense_46/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_46/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1

;0
<1
 
?
xlayer_metrics
ymetrics
=trainable_variables
>	variables

zlayers
{non_trainable_variables
?regularization_losses
|layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

}0
~1
2
?3
F
0
1
2
3
4
5
6
7
	8

9
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
\
?
thresholds
?true_positives
?false_positives
?	variables
?	keras_api
\
?
thresholds
?true_positives
?false_negatives
?	variables
?	keras_api
v
?true_positives
?true_negatives
?false_positives
?false_negatives
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
ca
VARIABLE_VALUEtrue_positives_2=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?	variables
}
VARIABLE_VALUEAdam/conv2d_40/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_40/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_41/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_41/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_45/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_45/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_46/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_46/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_40/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_40/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_41/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_41/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_45/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_45/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_46/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_46/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv2d_40_inputPlaceholder*0
_output_shapes
:??????????*
dtype0*%
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_40_inputconv2d_40/kernelconv2d_40/biasconv2d_41/kernelconv2d_41/biasdense_45/kerneldense_45/biasdense_46/kerneldense_46/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_89744014
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_40/kernel/Read/ReadVariableOp"conv2d_40/bias/Read/ReadVariableOp$conv2d_41/kernel/Read/ReadVariableOp"conv2d_41/bias/Read/ReadVariableOp#dense_45/kernel/Read/ReadVariableOp!dense_45/bias/Read/ReadVariableOp#dense_46/kernel/Read/ReadVariableOp!dense_46/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_2/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp+Adam/conv2d_40/kernel/m/Read/ReadVariableOp)Adam/conv2d_40/bias/m/Read/ReadVariableOp+Adam/conv2d_41/kernel/m/Read/ReadVariableOp)Adam/conv2d_41/bias/m/Read/ReadVariableOp*Adam/dense_45/kernel/m/Read/ReadVariableOp(Adam/dense_45/bias/m/Read/ReadVariableOp*Adam/dense_46/kernel/m/Read/ReadVariableOp(Adam/dense_46/bias/m/Read/ReadVariableOp+Adam/conv2d_40/kernel/v/Read/ReadVariableOp)Adam/conv2d_40/bias/v/Read/ReadVariableOp+Adam/conv2d_41/kernel/v/Read/ReadVariableOp)Adam/conv2d_41/bias/v/Read/ReadVariableOp*Adam/dense_45/kernel/v/Read/ReadVariableOp(Adam/dense_45/bias/v/Read/ReadVariableOp*Adam/dense_46/kernel/v/Read/ReadVariableOp(Adam/dense_46/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_save_89744467
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_40/kernelconv2d_40/biasconv2d_41/kernelconv2d_41/biasdense_45/kerneldense_45/biasdense_46/kerneldense_46/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttrue_positivesfalse_positivestrue_positives_1false_negativestrue_positives_2true_negativesfalse_positives_1false_negatives_1Adam/conv2d_40/kernel/mAdam/conv2d_40/bias/mAdam/conv2d_41/kernel/mAdam/conv2d_41/bias/mAdam/dense_45/kernel/mAdam/dense_45/bias/mAdam/dense_46/kernel/mAdam/dense_46/bias/mAdam/conv2d_40/kernel/vAdam/conv2d_40/bias/vAdam/conv2d_41/kernel/vAdam/conv2d_41/bias/vAdam/dense_45/kernel/vAdam/dense_45/bias/vAdam/dense_46/kernel/vAdam/dense_46/bias/v*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference__traced_restore_89744594??
?
?
,__inference_conv2d_40_layer_call_fn_89744175

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_40_layer_call_and_return_conditional_losses_897436462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????(2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?.
?
B__inference_m_45_layer_call_and_return_conditional_losses_89743850
conv2d_40_input
conv2d_40_89743657
conv2d_40_89743659
conv2d_41_89743715
conv2d_41_89743717
dense_45_89743787
dense_45_89743789
dense_46_89743844
dense_46_89743846
identity??!conv2d_40/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall? dense_45/StatefulPartitionedCall? dense_46/StatefulPartitionedCall?"dropout_50/StatefulPartitionedCall?"dropout_51/StatefulPartitionedCall?"dropout_52/StatefulPartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallconv2d_40_inputconv2d_40_89743657conv2d_40_89743659*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_40_layer_call_and_return_conditional_losses_897436462#
!conv2d_40/StatefulPartitionedCall?
"dropout_50/StatefulPartitionedCallStatefulPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_50_layer_call_and_return_conditional_losses_897436742$
"dropout_50/StatefulPartitionedCall?
 max_pooling2d_40/PartitionedCallPartitionedCall+dropout_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????T(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_897436132"
 max_pooling2d_40/PartitionedCall?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_40/PartitionedCall:output:0conv2d_41_89743715conv2d_41_89743717*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_41_layer_call_and_return_conditional_losses_897437042#
!conv2d_41/StatefulPartitionedCall?
"dropout_51/StatefulPartitionedCallStatefulPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0#^dropout_50/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_51_layer_call_and_return_conditional_losses_897437322$
"dropout_51/StatefulPartitionedCall?
 max_pooling2d_41/PartitionedCallPartitionedCall+dropout_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_897436252"
 max_pooling2d_41/PartitionedCall?
flatten_30/PartitionedCallPartitionedCall)max_pooling2d_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_30_layer_call_and_return_conditional_losses_897437572
flatten_30/PartitionedCall?
 dense_45/StatefulPartitionedCallStatefulPartitionedCall#flatten_30/PartitionedCall:output:0dense_45_89743787dense_45_89743789*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_45_layer_call_and_return_conditional_losses_897437762"
 dense_45/StatefulPartitionedCall?
"dropout_52/StatefulPartitionedCallStatefulPartitionedCall)dense_45/StatefulPartitionedCall:output:0#^dropout_51/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_52_layer_call_and_return_conditional_losses_897438042$
"dropout_52/StatefulPartitionedCall?
 dense_46/StatefulPartitionedCallStatefulPartitionedCall+dropout_52/StatefulPartitionedCall:output:0dense_46_89743844dense_46_89743846*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_46_layer_call_and_return_conditional_losses_897438332"
 dense_46/StatefulPartitionedCall?
IdentityIdentity)dense_46/StatefulPartitionedCall:output:0"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall#^dropout_50/StatefulPartitionedCall#^dropout_51/StatefulPartitionedCall#^dropout_52/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall2H
"dropout_50/StatefulPartitionedCall"dropout_50/StatefulPartitionedCall2H
"dropout_51/StatefulPartitionedCall"dropout_51/StatefulPartitionedCall2H
"dropout_52/StatefulPartitionedCall"dropout_52/StatefulPartitionedCall:a ]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_40_input
?
O
3__inference_max_pooling2d_41_layer_call_fn_89743631

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_897436252
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
G__inference_conv2d_41_layer_call_and_return_conditional_losses_89744213

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(`*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I`*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I`2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????I`2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????I`2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????T(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????T(
 
_user_specified_nameinputs
?

?
G__inference_conv2d_40_layer_call_and_return_conditional_losses_89744166

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????(*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????(2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????(2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????(2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_dense_45_layer_call_fn_89744280

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_45_layer_call_and_return_conditional_losses_897437762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
G__inference_conv2d_41_layer_call_and_return_conditional_losses_89743704

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(`*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I`*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I`2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????I`2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????I`2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????T(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????T(
 
_user_specified_nameinputs
?)
?
B__inference_m_45_layer_call_and_return_conditional_losses_89743964

inputs
conv2d_40_89743937
conv2d_40_89743939
conv2d_41_89743944
conv2d_41_89743946
dense_45_89743952
dense_45_89743954
dense_46_89743958
dense_46_89743960
identity??!conv2d_40/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall? dense_45/StatefulPartitionedCall? dense_46/StatefulPartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_40_89743937conv2d_40_89743939*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_40_layer_call_and_return_conditional_losses_897436462#
!conv2d_40/StatefulPartitionedCall?
dropout_50/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_50_layer_call_and_return_conditional_losses_897436792
dropout_50/PartitionedCall?
 max_pooling2d_40/PartitionedCallPartitionedCall#dropout_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????T(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_897436132"
 max_pooling2d_40/PartitionedCall?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_40/PartitionedCall:output:0conv2d_41_89743944conv2d_41_89743946*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_41_layer_call_and_return_conditional_losses_897437042#
!conv2d_41/StatefulPartitionedCall?
dropout_51/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_51_layer_call_and_return_conditional_losses_897437372
dropout_51/PartitionedCall?
 max_pooling2d_41/PartitionedCallPartitionedCall#dropout_51/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_897436252"
 max_pooling2d_41/PartitionedCall?
flatten_30/PartitionedCallPartitionedCall)max_pooling2d_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_30_layer_call_and_return_conditional_losses_897437572
flatten_30/PartitionedCall?
 dense_45/StatefulPartitionedCallStatefulPartitionedCall#flatten_30/PartitionedCall:output:0dense_45_89743952dense_45_89743954*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_45_layer_call_and_return_conditional_losses_897437762"
 dense_45/StatefulPartitionedCall?
dropout_52/PartitionedCallPartitionedCall)dense_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_52_layer_call_and_return_conditional_losses_897438092
dropout_52/PartitionedCall?
 dense_46/StatefulPartitionedCallStatefulPartitionedCall#dropout_52/PartitionedCall:output:0dense_46_89743958dense_46_89743960*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_46_layer_call_and_return_conditional_losses_897438332"
 dense_46/StatefulPartitionedCall?
IdentityIdentity)dense_46/StatefulPartitionedCall:output:0"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
H__inference_dropout_52_layer_call_and_return_conditional_losses_89744292

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
H__inference_dropout_52_layer_call_and_return_conditional_losses_89743804

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?R
?
!__inference__traced_save_89744467
file_prefix/
+savev2_conv2d_40_kernel_read_readvariableop-
)savev2_conv2d_40_bias_read_readvariableop/
+savev2_conv2d_41_kernel_read_readvariableop-
)savev2_conv2d_41_bias_read_readvariableop.
*savev2_dense_45_kernel_read_readvariableop,
(savev2_dense_45_bias_read_readvariableop.
*savev2_dense_46_kernel_read_readvariableop,
(savev2_dense_46_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop.
*savev2_false_negatives_read_readvariableop/
+savev2_true_positives_2_read_readvariableop-
)savev2_true_negatives_read_readvariableop0
,savev2_false_positives_1_read_readvariableop0
,savev2_false_negatives_1_read_readvariableop6
2savev2_adam_conv2d_40_kernel_m_read_readvariableop4
0savev2_adam_conv2d_40_bias_m_read_readvariableop6
2savev2_adam_conv2d_41_kernel_m_read_readvariableop4
0savev2_adam_conv2d_41_bias_m_read_readvariableop5
1savev2_adam_dense_45_kernel_m_read_readvariableop3
/savev2_adam_dense_45_bias_m_read_readvariableop5
1savev2_adam_dense_46_kernel_m_read_readvariableop3
/savev2_adam_dense_46_bias_m_read_readvariableop6
2savev2_adam_conv2d_40_kernel_v_read_readvariableop4
0savev2_adam_conv2d_40_bias_v_read_readvariableop6
2savev2_adam_conv2d_41_kernel_v_read_readvariableop4
0savev2_adam_conv2d_41_bias_v_read_readvariableop5
1savev2_adam_dense_45_kernel_v_read_readvariableop3
/savev2_adam_dense_45_bias_v_read_readvariableop5
1savev2_adam_dense_46_kernel_v_read_readvariableop3
/savev2_adam_dense_46_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*?
value?B?(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_40_kernel_read_readvariableop)savev2_conv2d_40_bias_read_readvariableop+savev2_conv2d_41_kernel_read_readvariableop)savev2_conv2d_41_bias_read_readvariableop*savev2_dense_45_kernel_read_readvariableop(savev2_dense_45_bias_read_readvariableop*savev2_dense_46_kernel_read_readvariableop(savev2_dense_46_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop+savev2_true_positives_2_read_readvariableop)savev2_true_negatives_read_readvariableop,savev2_false_positives_1_read_readvariableop,savev2_false_negatives_1_read_readvariableop2savev2_adam_conv2d_40_kernel_m_read_readvariableop0savev2_adam_conv2d_40_bias_m_read_readvariableop2savev2_adam_conv2d_41_kernel_m_read_readvariableop0savev2_adam_conv2d_41_bias_m_read_readvariableop1savev2_adam_dense_45_kernel_m_read_readvariableop/savev2_adam_dense_45_bias_m_read_readvariableop1savev2_adam_dense_46_kernel_m_read_readvariableop/savev2_adam_dense_46_bias_m_read_readvariableop2savev2_adam_conv2d_40_kernel_v_read_readvariableop0savev2_adam_conv2d_40_bias_v_read_readvariableop2savev2_adam_conv2d_41_kernel_v_read_readvariableop0savev2_adam_conv2d_41_bias_v_read_readvariableop1savev2_adam_dense_45_kernel_v_read_readvariableop/savev2_adam_dense_45_bias_v_read_readvariableop1savev2_adam_dense_46_kernel_v_read_readvariableop/savev2_adam_dense_46_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :(:(:(`:`:
??:?:	?X:X: : : : : : : :::::?:?:?:?:(:(:(`:`:
??:?:	?X:X:(:(:(`:`:
??:?:	?X:X: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:(: 

_output_shapes
:(:,(
&
_output_shapes
:(`: 

_output_shapes
:`:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?X: 

_output_shapes
:X:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:,(
&
_output_shapes
:(: 

_output_shapes
:(:,(
&
_output_shapes
:(`: 

_output_shapes
:`:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?X: 

_output_shapes
:X:, (
&
_output_shapes
:(: !

_output_shapes
:(:,"(
&
_output_shapes
:(`: #

_output_shapes
:`:&$"
 
_output_shapes
:
??:!%

_output_shapes	
:?:%&!

_output_shapes
:	?X: '

_output_shapes
:X:(

_output_shapes
: 
?
?
,__inference_conv2d_41_layer_call_fn_89744222

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_41_layer_call_and_return_conditional_losses_897437042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????I`2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????T(::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????T(
 
_user_specified_nameinputs
?	
?
F__inference_dense_46_layer_call_and_return_conditional_losses_89743833

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?X*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:X*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????X2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_m_45_layer_call_fn_89743932
conv2d_40_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_40_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_m_45_layer_call_and_return_conditional_losses_897439132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_40_input
?
f
H__inference_dropout_52_layer_call_and_return_conditional_losses_89744297

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
-__inference_dropout_51_layer_call_fn_89744244

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_51_layer_call_and_return_conditional_losses_897437322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????I`2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????I`22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????I`
 
_user_specified_nameinputs
?
?
+__inference_dense_46_layer_call_fn_89744327

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_46_layer_call_and_return_conditional_losses_897438332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
H__inference_dropout_51_layer_call_and_return_conditional_losses_89744239

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????I`2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????I`2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????I`:W S
/
_output_shapes
:?????????I`
 
_user_specified_nameinputs
?
I
-__inference_dropout_51_layer_call_fn_89744249

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_51_layer_call_and_return_conditional_losses_897437372
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????I`2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????I`:W S
/
_output_shapes
:?????????I`
 
_user_specified_nameinputs
?
g
H__inference_dropout_50_layer_call_and_return_conditional_losses_89744187

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????(2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????(*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????(2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????(2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????(2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????(2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????(:X T
0
_output_shapes
:??????????(
 
_user_specified_nameinputs
?
O
3__inference_max_pooling2d_40_layer_call_fn_89743619

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_897436132
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?*
?
B__inference_m_45_layer_call_and_return_conditional_losses_89743880
conv2d_40_input
conv2d_40_89743853
conv2d_40_89743855
conv2d_41_89743860
conv2d_41_89743862
dense_45_89743868
dense_45_89743870
dense_46_89743874
dense_46_89743876
identity??!conv2d_40/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall? dense_45/StatefulPartitionedCall? dense_46/StatefulPartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallconv2d_40_inputconv2d_40_89743853conv2d_40_89743855*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_40_layer_call_and_return_conditional_losses_897436462#
!conv2d_40/StatefulPartitionedCall?
dropout_50/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_50_layer_call_and_return_conditional_losses_897436792
dropout_50/PartitionedCall?
 max_pooling2d_40/PartitionedCallPartitionedCall#dropout_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????T(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_897436132"
 max_pooling2d_40/PartitionedCall?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_40/PartitionedCall:output:0conv2d_41_89743860conv2d_41_89743862*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_41_layer_call_and_return_conditional_losses_897437042#
!conv2d_41/StatefulPartitionedCall?
dropout_51/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_51_layer_call_and_return_conditional_losses_897437372
dropout_51/PartitionedCall?
 max_pooling2d_41/PartitionedCallPartitionedCall#dropout_51/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_897436252"
 max_pooling2d_41/PartitionedCall?
flatten_30/PartitionedCallPartitionedCall)max_pooling2d_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_30_layer_call_and_return_conditional_losses_897437572
flatten_30/PartitionedCall?
 dense_45/StatefulPartitionedCallStatefulPartitionedCall#flatten_30/PartitionedCall:output:0dense_45_89743868dense_45_89743870*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_45_layer_call_and_return_conditional_losses_897437762"
 dense_45/StatefulPartitionedCall?
dropout_52/PartitionedCallPartitionedCall)dense_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_52_layer_call_and_return_conditional_losses_897438092
dropout_52/PartitionedCall?
 dense_46/StatefulPartitionedCallStatefulPartitionedCall#dropout_52/PartitionedCall:output:0dense_46_89743874dense_46_89743876*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_46_layer_call_and_return_conditional_losses_897438332"
 dense_46/StatefulPartitionedCall?
IdentityIdentity)dense_46/StatefulPartitionedCall:output:0"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall:a ]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_40_input
?
?
&__inference_signature_wrapper_89744014
conv2d_40_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_40_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__wrapped_model_897436072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_40_input
?6
?
#__inference__wrapped_model_89743607
conv2d_40_input1
-m_45_conv2d_40_conv2d_readvariableop_resource2
.m_45_conv2d_40_biasadd_readvariableop_resource1
-m_45_conv2d_41_conv2d_readvariableop_resource2
.m_45_conv2d_41_biasadd_readvariableop_resource0
,m_45_dense_45_matmul_readvariableop_resource1
-m_45_dense_45_biasadd_readvariableop_resource0
,m_45_dense_46_matmul_readvariableop_resource1
-m_45_dense_46_biasadd_readvariableop_resource
identity??%m_45/conv2d_40/BiasAdd/ReadVariableOp?$m_45/conv2d_40/Conv2D/ReadVariableOp?%m_45/conv2d_41/BiasAdd/ReadVariableOp?$m_45/conv2d_41/Conv2D/ReadVariableOp?$m_45/dense_45/BiasAdd/ReadVariableOp?#m_45/dense_45/MatMul/ReadVariableOp?$m_45/dense_46/BiasAdd/ReadVariableOp?#m_45/dense_46/MatMul/ReadVariableOp?
$m_45/conv2d_40/Conv2D/ReadVariableOpReadVariableOp-m_45_conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02&
$m_45/conv2d_40/Conv2D/ReadVariableOp?
m_45/conv2d_40/Conv2DConv2Dconv2d_40_input,m_45/conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????(*
paddingVALID*
strides
2
m_45/conv2d_40/Conv2D?
%m_45/conv2d_40/BiasAdd/ReadVariableOpReadVariableOp.m_45_conv2d_40_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02'
%m_45/conv2d_40/BiasAdd/ReadVariableOp?
m_45/conv2d_40/BiasAddBiasAddm_45/conv2d_40/Conv2D:output:0-m_45/conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????(2
m_45/conv2d_40/BiasAdd?
m_45/conv2d_40/ReluRelum_45/conv2d_40/BiasAdd:output:0*
T0*0
_output_shapes
:??????????(2
m_45/conv2d_40/Relu?
m_45/dropout_50/IdentityIdentity!m_45/conv2d_40/Relu:activations:0*
T0*0
_output_shapes
:??????????(2
m_45/dropout_50/Identity?
m_45/max_pooling2d_40/MaxPoolMaxPool!m_45/dropout_50/Identity:output:0*/
_output_shapes
:?????????T(*
ksize
*
paddingVALID*
strides
2
m_45/max_pooling2d_40/MaxPool?
$m_45/conv2d_41/Conv2D/ReadVariableOpReadVariableOp-m_45_conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:(`*
dtype02&
$m_45/conv2d_41/Conv2D/ReadVariableOp?
m_45/conv2d_41/Conv2DConv2D&m_45/max_pooling2d_40/MaxPool:output:0,m_45/conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I`*
paddingVALID*
strides
2
m_45/conv2d_41/Conv2D?
%m_45/conv2d_41/BiasAdd/ReadVariableOpReadVariableOp.m_45_conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02'
%m_45/conv2d_41/BiasAdd/ReadVariableOp?
m_45/conv2d_41/BiasAddBiasAddm_45/conv2d_41/Conv2D:output:0-m_45/conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I`2
m_45/conv2d_41/BiasAdd?
m_45/conv2d_41/ReluRelum_45/conv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:?????????I`2
m_45/conv2d_41/Relu?
m_45/dropout_51/IdentityIdentity!m_45/conv2d_41/Relu:activations:0*
T0*/
_output_shapes
:?????????I`2
m_45/dropout_51/Identity?
m_45/max_pooling2d_41/MaxPoolMaxPool!m_45/dropout_51/Identity:output:0*/
_output_shapes
:?????????$`*
ksize
*
paddingVALID*
strides
2
m_45/max_pooling2d_41/MaxPool
m_45/flatten_30/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
m_45/flatten_30/Const?
m_45/flatten_30/ReshapeReshape&m_45/max_pooling2d_41/MaxPool:output:0m_45/flatten_30/Const:output:0*
T0*(
_output_shapes
:??????????2
m_45/flatten_30/Reshape?
#m_45/dense_45/MatMul/ReadVariableOpReadVariableOp,m_45_dense_45_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#m_45/dense_45/MatMul/ReadVariableOp?
m_45/dense_45/MatMulMatMul m_45/flatten_30/Reshape:output:0+m_45/dense_45/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
m_45/dense_45/MatMul?
$m_45/dense_45/BiasAdd/ReadVariableOpReadVariableOp-m_45_dense_45_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$m_45/dense_45/BiasAdd/ReadVariableOp?
m_45/dense_45/BiasAddBiasAddm_45/dense_45/MatMul:product:0,m_45/dense_45/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
m_45/dense_45/BiasAdd?
m_45/dense_45/ReluRelum_45/dense_45/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
m_45/dense_45/Relu?
m_45/dropout_52/IdentityIdentity m_45/dense_45/Relu:activations:0*
T0*(
_output_shapes
:??????????2
m_45/dropout_52/Identity?
#m_45/dense_46/MatMul/ReadVariableOpReadVariableOp,m_45_dense_46_matmul_readvariableop_resource*
_output_shapes
:	?X*
dtype02%
#m_45/dense_46/MatMul/ReadVariableOp?
m_45/dense_46/MatMulMatMul!m_45/dropout_52/Identity:output:0+m_45/dense_46/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X2
m_45/dense_46/MatMul?
$m_45/dense_46/BiasAdd/ReadVariableOpReadVariableOp-m_45_dense_46_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02&
$m_45/dense_46/BiasAdd/ReadVariableOp?
m_45/dense_46/BiasAddBiasAddm_45/dense_46/MatMul:product:0,m_45/dense_46/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X2
m_45/dense_46/BiasAdd?
m_45/dense_46/SigmoidSigmoidm_45/dense_46/BiasAdd:output:0*
T0*'
_output_shapes
:?????????X2
m_45/dense_46/Sigmoid?
IdentityIdentitym_45/dense_46/Sigmoid:y:0&^m_45/conv2d_40/BiasAdd/ReadVariableOp%^m_45/conv2d_40/Conv2D/ReadVariableOp&^m_45/conv2d_41/BiasAdd/ReadVariableOp%^m_45/conv2d_41/Conv2D/ReadVariableOp%^m_45/dense_45/BiasAdd/ReadVariableOp$^m_45/dense_45/MatMul/ReadVariableOp%^m_45/dense_46/BiasAdd/ReadVariableOp$^m_45/dense_46/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::2N
%m_45/conv2d_40/BiasAdd/ReadVariableOp%m_45/conv2d_40/BiasAdd/ReadVariableOp2L
$m_45/conv2d_40/Conv2D/ReadVariableOp$m_45/conv2d_40/Conv2D/ReadVariableOp2N
%m_45/conv2d_41/BiasAdd/ReadVariableOp%m_45/conv2d_41/BiasAdd/ReadVariableOp2L
$m_45/conv2d_41/Conv2D/ReadVariableOp$m_45/conv2d_41/Conv2D/ReadVariableOp2L
$m_45/dense_45/BiasAdd/ReadVariableOp$m_45/dense_45/BiasAdd/ReadVariableOp2J
#m_45/dense_45/MatMul/ReadVariableOp#m_45/dense_45/MatMul/ReadVariableOp2L
$m_45/dense_46/BiasAdd/ReadVariableOp$m_45/dense_46/BiasAdd/ReadVariableOp2J
#m_45/dense_46/MatMul/ReadVariableOp#m_45/dense_46/MatMul/ReadVariableOp:a ]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_40_input
?.
?
B__inference_m_45_layer_call_and_return_conditional_losses_89743913

inputs
conv2d_40_89743886
conv2d_40_89743888
conv2d_41_89743893
conv2d_41_89743895
dense_45_89743901
dense_45_89743903
dense_46_89743907
dense_46_89743909
identity??!conv2d_40/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall? dense_45/StatefulPartitionedCall? dense_46/StatefulPartitionedCall?"dropout_50/StatefulPartitionedCall?"dropout_51/StatefulPartitionedCall?"dropout_52/StatefulPartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_40_89743886conv2d_40_89743888*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_40_layer_call_and_return_conditional_losses_897436462#
!conv2d_40/StatefulPartitionedCall?
"dropout_50/StatefulPartitionedCallStatefulPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_50_layer_call_and_return_conditional_losses_897436742$
"dropout_50/StatefulPartitionedCall?
 max_pooling2d_40/PartitionedCallPartitionedCall+dropout_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????T(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_897436132"
 max_pooling2d_40/PartitionedCall?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_40/PartitionedCall:output:0conv2d_41_89743893conv2d_41_89743895*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_41_layer_call_and_return_conditional_losses_897437042#
!conv2d_41/StatefulPartitionedCall?
"dropout_51/StatefulPartitionedCallStatefulPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0#^dropout_50/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_51_layer_call_and_return_conditional_losses_897437322$
"dropout_51/StatefulPartitionedCall?
 max_pooling2d_41/PartitionedCallPartitionedCall+dropout_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_897436252"
 max_pooling2d_41/PartitionedCall?
flatten_30/PartitionedCallPartitionedCall)max_pooling2d_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_30_layer_call_and_return_conditional_losses_897437572
flatten_30/PartitionedCall?
 dense_45/StatefulPartitionedCallStatefulPartitionedCall#flatten_30/PartitionedCall:output:0dense_45_89743901dense_45_89743903*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_45_layer_call_and_return_conditional_losses_897437762"
 dense_45/StatefulPartitionedCall?
"dropout_52/StatefulPartitionedCallStatefulPartitionedCall)dense_45/StatefulPartitionedCall:output:0#^dropout_51/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_52_layer_call_and_return_conditional_losses_897438042$
"dropout_52/StatefulPartitionedCall?
 dense_46/StatefulPartitionedCallStatefulPartitionedCall+dropout_52/StatefulPartitionedCall:output:0dense_46_89743907dense_46_89743909*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_46_layer_call_and_return_conditional_losses_897438332"
 dense_46/StatefulPartitionedCall?
IdentityIdentity)dense_46/StatefulPartitionedCall:output:0"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall#^dropout_50/StatefulPartitionedCall#^dropout_51/StatefulPartitionedCall#^dropout_52/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall2H
"dropout_50/StatefulPartitionedCall"dropout_50/StatefulPartitionedCall2H
"dropout_51/StatefulPartitionedCall"dropout_51/StatefulPartitionedCall2H
"dropout_52/StatefulPartitionedCall"dropout_52/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
H__inference_dropout_51_layer_call_and_return_conditional_losses_89743737

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????I`2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????I`2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????I`:W S
/
_output_shapes
:?????????I`
 
_user_specified_nameinputs
?	
?
F__inference_dense_45_layer_call_and_return_conditional_losses_89744271

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
-__inference_dropout_52_layer_call_fn_89744302

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_52_layer_call_and_return_conditional_losses_897438042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
-__inference_dropout_50_layer_call_fn_89744197

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_50_layer_call_and_return_conditional_losses_897436742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????(2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????(22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????(
 
_user_specified_nameinputs
?
f
H__inference_dropout_50_layer_call_and_return_conditional_losses_89744192

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????(2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????(2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????(:X T
0
_output_shapes
:??????????(
 
_user_specified_nameinputs
?1
?
B__inference_m_45_layer_call_and_return_conditional_losses_89744113

inputs,
(conv2d_40_conv2d_readvariableop_resource-
)conv2d_40_biasadd_readvariableop_resource,
(conv2d_41_conv2d_readvariableop_resource-
)conv2d_41_biasadd_readvariableop_resource+
'dense_45_matmul_readvariableop_resource,
(dense_45_biasadd_readvariableop_resource+
'dense_46_matmul_readvariableop_resource,
(dense_46_biasadd_readvariableop_resource
identity?? conv2d_40/BiasAdd/ReadVariableOp?conv2d_40/Conv2D/ReadVariableOp? conv2d_41/BiasAdd/ReadVariableOp?conv2d_41/Conv2D/ReadVariableOp?dense_45/BiasAdd/ReadVariableOp?dense_45/MatMul/ReadVariableOp?dense_46/BiasAdd/ReadVariableOp?dense_46/MatMul/ReadVariableOp?
conv2d_40/Conv2D/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02!
conv2d_40/Conv2D/ReadVariableOp?
conv2d_40/Conv2DConv2Dinputs'conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????(*
paddingVALID*
strides
2
conv2d_40/Conv2D?
 conv2d_40/BiasAdd/ReadVariableOpReadVariableOp)conv2d_40_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02"
 conv2d_40/BiasAdd/ReadVariableOp?
conv2d_40/BiasAddBiasAddconv2d_40/Conv2D:output:0(conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????(2
conv2d_40/BiasAdd
conv2d_40/ReluReluconv2d_40/BiasAdd:output:0*
T0*0
_output_shapes
:??????????(2
conv2d_40/Relu?
dropout_50/IdentityIdentityconv2d_40/Relu:activations:0*
T0*0
_output_shapes
:??????????(2
dropout_50/Identity?
max_pooling2d_40/MaxPoolMaxPooldropout_50/Identity:output:0*/
_output_shapes
:?????????T(*
ksize
*
paddingVALID*
strides
2
max_pooling2d_40/MaxPool?
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:(`*
dtype02!
conv2d_41/Conv2D/ReadVariableOp?
conv2d_41/Conv2DConv2D!max_pooling2d_40/MaxPool:output:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I`*
paddingVALID*
strides
2
conv2d_41/Conv2D?
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02"
 conv2d_41/BiasAdd/ReadVariableOp?
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I`2
conv2d_41/BiasAdd~
conv2d_41/ReluReluconv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:?????????I`2
conv2d_41/Relu?
dropout_51/IdentityIdentityconv2d_41/Relu:activations:0*
T0*/
_output_shapes
:?????????I`2
dropout_51/Identity?
max_pooling2d_41/MaxPoolMaxPooldropout_51/Identity:output:0*/
_output_shapes
:?????????$`*
ksize
*
paddingVALID*
strides
2
max_pooling2d_41/MaxPoolu
flatten_30/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_30/Const?
flatten_30/ReshapeReshape!max_pooling2d_41/MaxPool:output:0flatten_30/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_30/Reshape?
dense_45/MatMul/ReadVariableOpReadVariableOp'dense_45_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_45/MatMul/ReadVariableOp?
dense_45/MatMulMatMulflatten_30/Reshape:output:0&dense_45/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_45/MatMul?
dense_45/BiasAdd/ReadVariableOpReadVariableOp(dense_45_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_45/BiasAdd/ReadVariableOp?
dense_45/BiasAddBiasAdddense_45/MatMul:product:0'dense_45/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_45/BiasAddt
dense_45/ReluReludense_45/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_45/Relu?
dropout_52/IdentityIdentitydense_45/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_52/Identity?
dense_46/MatMul/ReadVariableOpReadVariableOp'dense_46_matmul_readvariableop_resource*
_output_shapes
:	?X*
dtype02 
dense_46/MatMul/ReadVariableOp?
dense_46/MatMulMatMuldropout_52/Identity:output:0&dense_46/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X2
dense_46/MatMul?
dense_46/BiasAdd/ReadVariableOpReadVariableOp(dense_46_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02!
dense_46/BiasAdd/ReadVariableOp?
dense_46/BiasAddBiasAdddense_46/MatMul:product:0'dense_46/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X2
dense_46/BiasAdd|
dense_46/SigmoidSigmoiddense_46/BiasAdd:output:0*
T0*'
_output_shapes
:?????????X2
dense_46/Sigmoid?
IdentityIdentitydense_46/Sigmoid:y:0!^conv2d_40/BiasAdd/ReadVariableOp ^conv2d_40/Conv2D/ReadVariableOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp ^dense_45/BiasAdd/ReadVariableOp^dense_45/MatMul/ReadVariableOp ^dense_46/BiasAdd/ReadVariableOp^dense_46/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::2D
 conv2d_40/BiasAdd/ReadVariableOp conv2d_40/BiasAdd/ReadVariableOp2B
conv2d_40/Conv2D/ReadVariableOpconv2d_40/Conv2D/ReadVariableOp2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2B
dense_45/BiasAdd/ReadVariableOpdense_45/BiasAdd/ReadVariableOp2@
dense_45/MatMul/ReadVariableOpdense_45/MatMul/ReadVariableOp2B
dense_46/BiasAdd/ReadVariableOpdense_46/BiasAdd/ReadVariableOp2@
dense_46/MatMul/ReadVariableOpdense_46/MatMul/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
I
-__inference_dropout_52_layer_call_fn_89744307

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_52_layer_call_and_return_conditional_losses_897438092
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_m_45_layer_call_fn_89744134

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_m_45_layer_call_and_return_conditional_losses_897439132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_flatten_30_layer_call_and_return_conditional_losses_89744255

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????$`:W S
/
_output_shapes
:?????????$`
 
_user_specified_nameinputs
?	
?
F__inference_dense_46_layer_call_and_return_conditional_losses_89744318

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?X*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:X*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????X2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
$__inference__traced_restore_89744594
file_prefix%
!assignvariableop_conv2d_40_kernel%
!assignvariableop_1_conv2d_40_bias'
#assignvariableop_2_conv2d_41_kernel%
!assignvariableop_3_conv2d_41_bias&
"assignvariableop_4_dense_45_kernel$
 assignvariableop_5_dense_45_bias&
"assignvariableop_6_dense_46_kernel$
 assignvariableop_7_dense_46_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count&
"assignvariableop_15_true_positives'
#assignvariableop_16_false_positives(
$assignvariableop_17_true_positives_1'
#assignvariableop_18_false_negatives(
$assignvariableop_19_true_positives_2&
"assignvariableop_20_true_negatives)
%assignvariableop_21_false_positives_1)
%assignvariableop_22_false_negatives_1/
+assignvariableop_23_adam_conv2d_40_kernel_m-
)assignvariableop_24_adam_conv2d_40_bias_m/
+assignvariableop_25_adam_conv2d_41_kernel_m-
)assignvariableop_26_adam_conv2d_41_bias_m.
*assignvariableop_27_adam_dense_45_kernel_m,
(assignvariableop_28_adam_dense_45_bias_m.
*assignvariableop_29_adam_dense_46_kernel_m,
(assignvariableop_30_adam_dense_46_bias_m/
+assignvariableop_31_adam_conv2d_40_kernel_v-
)assignvariableop_32_adam_conv2d_40_bias_v/
+assignvariableop_33_adam_conv2d_41_kernel_v-
)assignvariableop_34_adam_conv2d_41_bias_v.
*assignvariableop_35_adam_dense_45_kernel_v,
(assignvariableop_36_adam_dense_45_bias_v.
*assignvariableop_37_adam_dense_46_kernel_v,
(assignvariableop_38_adam_dense_46_bias_v
identity_40??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*?
value?B?(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_40_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_40_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_41_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_41_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_45_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_45_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_46_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_46_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp"assignvariableop_15_true_positivesIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_false_positivesIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp$assignvariableop_17_true_positives_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_false_negativesIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp$assignvariableop_19_true_positives_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp"assignvariableop_20_true_negativesIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp%assignvariableop_21_false_positives_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp%assignvariableop_22_false_negatives_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_40_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_40_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_41_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_41_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_45_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_45_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_46_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_46_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_40_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_40_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_41_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_41_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_45_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_45_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_46_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_46_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_39?
Identity_40IdentityIdentity_39:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_40"#
identity_40Identity_40:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
I
-__inference_dropout_50_layer_call_fn_89744202

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_50_layer_call_and_return_conditional_losses_897436792
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????(2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????(:X T
0
_output_shapes
:??????????(
 
_user_specified_nameinputs
?N
?
B__inference_m_45_layer_call_and_return_conditional_losses_89744074

inputs,
(conv2d_40_conv2d_readvariableop_resource-
)conv2d_40_biasadd_readvariableop_resource,
(conv2d_41_conv2d_readvariableop_resource-
)conv2d_41_biasadd_readvariableop_resource+
'dense_45_matmul_readvariableop_resource,
(dense_45_biasadd_readvariableop_resource+
'dense_46_matmul_readvariableop_resource,
(dense_46_biasadd_readvariableop_resource
identity?? conv2d_40/BiasAdd/ReadVariableOp?conv2d_40/Conv2D/ReadVariableOp? conv2d_41/BiasAdd/ReadVariableOp?conv2d_41/Conv2D/ReadVariableOp?dense_45/BiasAdd/ReadVariableOp?dense_45/MatMul/ReadVariableOp?dense_46/BiasAdd/ReadVariableOp?dense_46/MatMul/ReadVariableOp?
conv2d_40/Conv2D/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02!
conv2d_40/Conv2D/ReadVariableOp?
conv2d_40/Conv2DConv2Dinputs'conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????(*
paddingVALID*
strides
2
conv2d_40/Conv2D?
 conv2d_40/BiasAdd/ReadVariableOpReadVariableOp)conv2d_40_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02"
 conv2d_40/BiasAdd/ReadVariableOp?
conv2d_40/BiasAddBiasAddconv2d_40/Conv2D:output:0(conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????(2
conv2d_40/BiasAdd
conv2d_40/ReluReluconv2d_40/BiasAdd:output:0*
T0*0
_output_shapes
:??????????(2
conv2d_40/Reluy
dropout_50/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_50/dropout/Const?
dropout_50/dropout/MulMulconv2d_40/Relu:activations:0!dropout_50/dropout/Const:output:0*
T0*0
_output_shapes
:??????????(2
dropout_50/dropout/Mul?
dropout_50/dropout/ShapeShapeconv2d_40/Relu:activations:0*
T0*
_output_shapes
:2
dropout_50/dropout/Shape?
/dropout_50/dropout/random_uniform/RandomUniformRandomUniform!dropout_50/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????(*
dtype021
/dropout_50/dropout/random_uniform/RandomUniform?
!dropout_50/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_50/dropout/GreaterEqual/y?
dropout_50/dropout/GreaterEqualGreaterEqual8dropout_50/dropout/random_uniform/RandomUniform:output:0*dropout_50/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????(2!
dropout_50/dropout/GreaterEqual?
dropout_50/dropout/CastCast#dropout_50/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????(2
dropout_50/dropout/Cast?
dropout_50/dropout/Mul_1Muldropout_50/dropout/Mul:z:0dropout_50/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????(2
dropout_50/dropout/Mul_1?
max_pooling2d_40/MaxPoolMaxPooldropout_50/dropout/Mul_1:z:0*/
_output_shapes
:?????????T(*
ksize
*
paddingVALID*
strides
2
max_pooling2d_40/MaxPool?
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:(`*
dtype02!
conv2d_41/Conv2D/ReadVariableOp?
conv2d_41/Conv2DConv2D!max_pooling2d_40/MaxPool:output:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I`*
paddingVALID*
strides
2
conv2d_41/Conv2D?
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02"
 conv2d_41/BiasAdd/ReadVariableOp?
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I`2
conv2d_41/BiasAdd~
conv2d_41/ReluReluconv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:?????????I`2
conv2d_41/Reluy
dropout_51/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_51/dropout/Const?
dropout_51/dropout/MulMulconv2d_41/Relu:activations:0!dropout_51/dropout/Const:output:0*
T0*/
_output_shapes
:?????????I`2
dropout_51/dropout/Mul?
dropout_51/dropout/ShapeShapeconv2d_41/Relu:activations:0*
T0*
_output_shapes
:2
dropout_51/dropout/Shape?
/dropout_51/dropout/random_uniform/RandomUniformRandomUniform!dropout_51/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????I`*
dtype021
/dropout_51/dropout/random_uniform/RandomUniform?
!dropout_51/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_51/dropout/GreaterEqual/y?
dropout_51/dropout/GreaterEqualGreaterEqual8dropout_51/dropout/random_uniform/RandomUniform:output:0*dropout_51/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????I`2!
dropout_51/dropout/GreaterEqual?
dropout_51/dropout/CastCast#dropout_51/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????I`2
dropout_51/dropout/Cast?
dropout_51/dropout/Mul_1Muldropout_51/dropout/Mul:z:0dropout_51/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????I`2
dropout_51/dropout/Mul_1?
max_pooling2d_41/MaxPoolMaxPooldropout_51/dropout/Mul_1:z:0*/
_output_shapes
:?????????$`*
ksize
*
paddingVALID*
strides
2
max_pooling2d_41/MaxPoolu
flatten_30/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_30/Const?
flatten_30/ReshapeReshape!max_pooling2d_41/MaxPool:output:0flatten_30/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_30/Reshape?
dense_45/MatMul/ReadVariableOpReadVariableOp'dense_45_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_45/MatMul/ReadVariableOp?
dense_45/MatMulMatMulflatten_30/Reshape:output:0&dense_45/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_45/MatMul?
dense_45/BiasAdd/ReadVariableOpReadVariableOp(dense_45_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_45/BiasAdd/ReadVariableOp?
dense_45/BiasAddBiasAdddense_45/MatMul:product:0'dense_45/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_45/BiasAddt
dense_45/ReluReludense_45/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_45/Reluy
dropout_52/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_52/dropout/Const?
dropout_52/dropout/MulMuldense_45/Relu:activations:0!dropout_52/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_52/dropout/Mul
dropout_52/dropout/ShapeShapedense_45/Relu:activations:0*
T0*
_output_shapes
:2
dropout_52/dropout/Shape?
/dropout_52/dropout/random_uniform/RandomUniformRandomUniform!dropout_52/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype021
/dropout_52/dropout/random_uniform/RandomUniform?
!dropout_52/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_52/dropout/GreaterEqual/y?
dropout_52/dropout/GreaterEqualGreaterEqual8dropout_52/dropout/random_uniform/RandomUniform:output:0*dropout_52/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
dropout_52/dropout/GreaterEqual?
dropout_52/dropout/CastCast#dropout_52/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_52/dropout/Cast?
dropout_52/dropout/Mul_1Muldropout_52/dropout/Mul:z:0dropout_52/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_52/dropout/Mul_1?
dense_46/MatMul/ReadVariableOpReadVariableOp'dense_46_matmul_readvariableop_resource*
_output_shapes
:	?X*
dtype02 
dense_46/MatMul/ReadVariableOp?
dense_46/MatMulMatMuldropout_52/dropout/Mul_1:z:0&dense_46/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X2
dense_46/MatMul?
dense_46/BiasAdd/ReadVariableOpReadVariableOp(dense_46_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02!
dense_46/BiasAdd/ReadVariableOp?
dense_46/BiasAddBiasAdddense_46/MatMul:product:0'dense_46/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X2
dense_46/BiasAdd|
dense_46/SigmoidSigmoiddense_46/BiasAdd:output:0*
T0*'
_output_shapes
:?????????X2
dense_46/Sigmoid?
IdentityIdentitydense_46/Sigmoid:y:0!^conv2d_40/BiasAdd/ReadVariableOp ^conv2d_40/Conv2D/ReadVariableOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp ^dense_45/BiasAdd/ReadVariableOp^dense_45/MatMul/ReadVariableOp ^dense_46/BiasAdd/ReadVariableOp^dense_46/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::2D
 conv2d_40/BiasAdd/ReadVariableOp conv2d_40/BiasAdd/ReadVariableOp2B
conv2d_40/Conv2D/ReadVariableOpconv2d_40/Conv2D/ReadVariableOp2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2B
dense_45/BiasAdd/ReadVariableOpdense_45/BiasAdd/ReadVariableOp2@
dense_45/MatMul/ReadVariableOpdense_45/MatMul/ReadVariableOp2B
dense_46/BiasAdd/ReadVariableOpdense_46/BiasAdd/ReadVariableOp2@
dense_46/MatMul/ReadVariableOpdense_46/MatMul/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_m_45_layer_call_fn_89744155

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_m_45_layer_call_and_return_conditional_losses_897439642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_45_layer_call_and_return_conditional_losses_89743776

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
H__inference_dropout_50_layer_call_and_return_conditional_losses_89743674

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????(2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????(*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????(2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????(2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????(2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????(2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????(:X T
0
_output_shapes
:??????????(
 
_user_specified_nameinputs
?

?
G__inference_conv2d_40_layer_call_and_return_conditional_losses_89743646

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:(*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????(*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????(2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????(2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????(2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
j
N__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_89743625

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
I
-__inference_flatten_30_layer_call_fn_89744260

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_30_layer_call_and_return_conditional_losses_897437572
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????$`:W S
/
_output_shapes
:?????????$`
 
_user_specified_nameinputs
?
f
H__inference_dropout_50_layer_call_and_return_conditional_losses_89743679

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????(2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????(2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????(:X T
0
_output_shapes
:??????????(
 
_user_specified_nameinputs
?
j
N__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_89743613

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
'__inference_m_45_layer_call_fn_89743983
conv2d_40_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_40_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_m_45_layer_call_and_return_conditional_losses_897439642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_40_input
?
g
H__inference_dropout_51_layer_call_and_return_conditional_losses_89743732

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????I`2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????I`*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????I`2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????I`2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????I`2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????I`2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????I`:W S
/
_output_shapes
:?????????I`
 
_user_specified_nameinputs
?
d
H__inference_flatten_30_layer_call_and_return_conditional_losses_89743757

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????$`:W S
/
_output_shapes
:?????????$`
 
_user_specified_nameinputs
?
f
H__inference_dropout_52_layer_call_and_return_conditional_losses_89743809

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
H__inference_dropout_51_layer_call_and_return_conditional_losses_89744234

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????I`2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????I`*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????I`2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????I`2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????I`2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????I`2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????I`:W S
/
_output_shapes
:?????????I`
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
T
conv2d_40_inputA
!serving_default_conv2d_40_input:0??????????<
dense_460
StatefulPartitionedCall:0?????????Xtensorflow/serving/predict:??
?g
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"?d
_tf_keras_sequential?c{"class_name": "Sequential", "name": "m_45", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "m_45", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_40_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_40", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "filters": 40, "kernel_size": {"class_name": "__tuple__", "items": [24, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_50", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_40", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [12, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_51", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_41", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_30", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_45", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_52", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_46", "trainable": true, "dtype": "float32", "units": 88, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192, 5, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "m_45", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_40_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_40", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "filters": 40, "kernel_size": {"class_name": "__tuple__", "items": [24, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_50", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_40", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [12, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_51", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_41", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_30", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_45", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_52", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_46", "trainable": true, "dtype": "float32", "units": 88, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": [[{"class_name": "Precision", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "Recall", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0006000000284984708, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_40", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "filters": 40, "kernel_size": {"class_name": "__tuple__", "items": [24, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192, 5, 1]}}
?
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_50", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_50", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_40", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

kernel
 bias
!trainable_variables
"	variables
#regularization_losses
$	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [12, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 84, 3, 40]}}
?
%trainable_variables
&	variables
'regularization_losses
(	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_51", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_51", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?
)trainable_variables
*	variables
+regularization_losses
,	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_41", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
-trainable_variables
.	variables
/regularization_losses
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_30", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

1kernel
2bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_45", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_45", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3456}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3456]}}
?
7trainable_variables
8	variables
9regularization_losses
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_52", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_52", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?

;kernel
<bias
=trainable_variables
>	variables
?regularization_losses
@	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_46", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_46", "trainable": true, "dtype": "float32", "units": 88, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_ratem?m?m? m?1m?2m?;m?<m?v?v?v? v?1v?2v?;v?<v?"
	optimizer
X
0
1
2
 3
14
25
;6
<7"
trackable_list_wrapper
X
0
1
2
 3
14
25
;6
<7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Flayer_metrics
Gmetrics
trainable_variables
	variables

Hlayers
Inon_trainable_variables
regularization_losses
Jlayer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:((2conv2d_40/kernel
:(2conv2d_40/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Klayer_metrics
Lmetrics
trainable_variables
	variables

Mlayers
Nnon_trainable_variables
regularization_losses
Olayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Player_metrics
Qmetrics
trainable_variables
	variables

Rlayers
Snon_trainable_variables
regularization_losses
Tlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ulayer_metrics
Vmetrics
trainable_variables
	variables

Wlayers
Xnon_trainable_variables
regularization_losses
Ylayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:((`2conv2d_41/kernel
:`2conv2d_41/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Zlayer_metrics
[metrics
!trainable_variables
"	variables

\layers
]non_trainable_variables
#regularization_losses
^layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
_layer_metrics
`metrics
%trainable_variables
&	variables

alayers
bnon_trainable_variables
'regularization_losses
clayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
dlayer_metrics
emetrics
)trainable_variables
*	variables

flayers
gnon_trainable_variables
+regularization_losses
hlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
ilayer_metrics
jmetrics
-trainable_variables
.	variables

klayers
lnon_trainable_variables
/regularization_losses
mlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_45/kernel
:?2dense_45/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
nlayer_metrics
ometrics
3trainable_variables
4	variables

players
qnon_trainable_variables
5regularization_losses
rlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
slayer_metrics
tmetrics
7trainable_variables
8	variables

ulayers
vnon_trainable_variables
9regularization_losses
wlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?X2dense_46/kernel
:X2dense_46/bias
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
xlayer_metrics
ymetrics
=trainable_variables
>	variables

zlayers
{non_trainable_variables
?regularization_losses
|layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
=
}0
~1
2
?3"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
?
thresholds
?true_positives
?false_positives
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Precision", "name": "precision", "dtype": "float32", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
?
?
thresholds
?true_positives
?false_negatives
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Recall", "name": "recall", "dtype": "float32", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
?"
?true_positives
?true_negatives
?false_positives
?false_negatives
?	variables
?	keras_api"?!
_tf_keras_metric?!{"class_name": "AUC", "name": "auc", "dtype": "float32", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:? (2true_positives
:? (2true_negatives
 :? (2false_positives
 :? (2false_negatives
@
?0
?1
?2
?3"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:-(2Adam/conv2d_40/kernel/m
!:(2Adam/conv2d_40/bias/m
/:-(`2Adam/conv2d_41/kernel/m
!:`2Adam/conv2d_41/bias/m
(:&
??2Adam/dense_45/kernel/m
!:?2Adam/dense_45/bias/m
':%	?X2Adam/dense_46/kernel/m
 :X2Adam/dense_46/bias/m
/:-(2Adam/conv2d_40/kernel/v
!:(2Adam/conv2d_40/bias/v
/:-(`2Adam/conv2d_41/kernel/v
!:`2Adam/conv2d_41/bias/v
(:&
??2Adam/dense_45/kernel/v
!:?2Adam/dense_45/bias/v
':%	?X2Adam/dense_46/kernel/v
 :X2Adam/dense_46/bias/v
?2?
B__inference_m_45_layer_call_and_return_conditional_losses_89743880
B__inference_m_45_layer_call_and_return_conditional_losses_89743850
B__inference_m_45_layer_call_and_return_conditional_losses_89744113
B__inference_m_45_layer_call_and_return_conditional_losses_89744074?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_m_45_layer_call_fn_89743983
'__inference_m_45_layer_call_fn_89743932
'__inference_m_45_layer_call_fn_89744155
'__inference_m_45_layer_call_fn_89744134?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference__wrapped_model_89743607?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/
conv2d_40_input??????????
?2?
G__inference_conv2d_40_layer_call_and_return_conditional_losses_89744166?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_conv2d_40_layer_call_fn_89744175?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dropout_50_layer_call_and_return_conditional_losses_89744192
H__inference_dropout_50_layer_call_and_return_conditional_losses_89744187?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_dropout_50_layer_call_fn_89744197
-__inference_dropout_50_layer_call_fn_89744202?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_89743613?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
3__inference_max_pooling2d_40_layer_call_fn_89743619?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_conv2d_41_layer_call_and_return_conditional_losses_89744213?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_conv2d_41_layer_call_fn_89744222?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dropout_51_layer_call_and_return_conditional_losses_89744234
H__inference_dropout_51_layer_call_and_return_conditional_losses_89744239?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_dropout_51_layer_call_fn_89744244
-__inference_dropout_51_layer_call_fn_89744249?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_89743625?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
3__inference_max_pooling2d_41_layer_call_fn_89743631?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
H__inference_flatten_30_layer_call_and_return_conditional_losses_89744255?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_flatten_30_layer_call_fn_89744260?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_45_layer_call_and_return_conditional_losses_89744271?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_45_layer_call_fn_89744280?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dropout_52_layer_call_and_return_conditional_losses_89744297
H__inference_dropout_52_layer_call_and_return_conditional_losses_89744292?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_dropout_52_layer_call_fn_89744302
-__inference_dropout_52_layer_call_fn_89744307?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dense_46_layer_call_and_return_conditional_losses_89744318?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_46_layer_call_fn_89744327?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_89744014conv2d_40_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_89743607? 12;<A?>
7?4
2?/
conv2d_40_input??????????
? "3?0
.
dense_46"?
dense_46?????????X?
G__inference_conv2d_40_layer_call_and_return_conditional_losses_89744166n8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????(
? ?
,__inference_conv2d_40_layer_call_fn_89744175a8?5
.?+
)?&
inputs??????????
? "!???????????(?
G__inference_conv2d_41_layer_call_and_return_conditional_losses_89744213l 7?4
-?*
(?%
inputs?????????T(
? "-?*
#? 
0?????????I`
? ?
,__inference_conv2d_41_layer_call_fn_89744222_ 7?4
-?*
(?%
inputs?????????T(
? " ??????????I`?
F__inference_dense_45_layer_call_and_return_conditional_losses_89744271^120?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_45_layer_call_fn_89744280Q120?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_46_layer_call_and_return_conditional_losses_89744318];<0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????X
? 
+__inference_dense_46_layer_call_fn_89744327P;<0?-
&?#
!?
inputs??????????
? "??????????X?
H__inference_dropout_50_layer_call_and_return_conditional_losses_89744187n<?9
2?/
)?&
inputs??????????(
p
? ".?+
$?!
0??????????(
? ?
H__inference_dropout_50_layer_call_and_return_conditional_losses_89744192n<?9
2?/
)?&
inputs??????????(
p 
? ".?+
$?!
0??????????(
? ?
-__inference_dropout_50_layer_call_fn_89744197a<?9
2?/
)?&
inputs??????????(
p
? "!???????????(?
-__inference_dropout_50_layer_call_fn_89744202a<?9
2?/
)?&
inputs??????????(
p 
? "!???????????(?
H__inference_dropout_51_layer_call_and_return_conditional_losses_89744234l;?8
1?.
(?%
inputs?????????I`
p
? "-?*
#? 
0?????????I`
? ?
H__inference_dropout_51_layer_call_and_return_conditional_losses_89744239l;?8
1?.
(?%
inputs?????????I`
p 
? "-?*
#? 
0?????????I`
? ?
-__inference_dropout_51_layer_call_fn_89744244_;?8
1?.
(?%
inputs?????????I`
p
? " ??????????I`?
-__inference_dropout_51_layer_call_fn_89744249_;?8
1?.
(?%
inputs?????????I`
p 
? " ??????????I`?
H__inference_dropout_52_layer_call_and_return_conditional_losses_89744292^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
H__inference_dropout_52_layer_call_and_return_conditional_losses_89744297^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
-__inference_dropout_52_layer_call_fn_89744302Q4?1
*?'
!?
inputs??????????
p
? "????????????
-__inference_dropout_52_layer_call_fn_89744307Q4?1
*?'
!?
inputs??????????
p 
? "????????????
H__inference_flatten_30_layer_call_and_return_conditional_losses_89744255a7?4
-?*
(?%
inputs?????????$`
? "&?#
?
0??????????
? ?
-__inference_flatten_30_layer_call_fn_89744260T7?4
-?*
(?%
inputs?????????$`
? "????????????
B__inference_m_45_layer_call_and_return_conditional_losses_89743850| 12;<I?F
??<
2?/
conv2d_40_input??????????
p

 
? "%?"
?
0?????????X
? ?
B__inference_m_45_layer_call_and_return_conditional_losses_89743880| 12;<I?F
??<
2?/
conv2d_40_input??????????
p 

 
? "%?"
?
0?????????X
? ?
B__inference_m_45_layer_call_and_return_conditional_losses_89744074s 12;<@?=
6?3
)?&
inputs??????????
p

 
? "%?"
?
0?????????X
? ?
B__inference_m_45_layer_call_and_return_conditional_losses_89744113s 12;<@?=
6?3
)?&
inputs??????????
p 

 
? "%?"
?
0?????????X
? ?
'__inference_m_45_layer_call_fn_89743932o 12;<I?F
??<
2?/
conv2d_40_input??????????
p

 
? "??????????X?
'__inference_m_45_layer_call_fn_89743983o 12;<I?F
??<
2?/
conv2d_40_input??????????
p 

 
? "??????????X?
'__inference_m_45_layer_call_fn_89744134f 12;<@?=
6?3
)?&
inputs??????????
p

 
? "??????????X?
'__inference_m_45_layer_call_fn_89744155f 12;<@?=
6?3
)?&
inputs??????????
p 

 
? "??????????X?
N__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_89743613?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
3__inference_max_pooling2d_40_layer_call_fn_89743619?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
N__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_89743625?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
3__inference_max_pooling2d_41_layer_call_fn_89743631?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
&__inference_signature_wrapper_89744014? 12;<T?Q
? 
J?G
E
conv2d_40_input2?/
conv2d_40_input??????????"3?0
.
dense_46"?
dense_46?????????X