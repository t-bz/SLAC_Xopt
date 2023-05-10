import torch
from typing import Optional, Union
from gpytorch.priors import Prior, NormalPrior, GammaPrior
from gpytorch.means import ConstantMean
from gpytorch.means.mean import Mean
from gpytorch.constraints import Interval, Positive


class CustomMean(Mean):
    def __init__(
            self,
            model: torch.nn.Module,
            **kwargs
    ):
        """Custom prior mean for a GP based on an arbitrary model.

        Args:
            model: Representation of the model.

        Attributes:
            config (dict): Stores the given keyword arguments.
        """
        super().__init__()
        self.model = model
        self.config = kwargs

    def forward(self, x):
        return self.model(x)


class InputOffsetCalibration(CustomMean):
    def __init__(
            self,
            model: torch.nn.Module,
            **kwargs
    ):
        """Prior mean with learnable input offset calibration.

        Inputs are offset by a learnable constant parameter:
        y = model(x + x_shift).

        Args:
            model: Inherited from CustomMean.

        Keyword Args:
            x_shift_dim (int): Dimension of the x_shift parameter.
              Defaults to 1.
            x_shift_prior (Optional[Prior]): Prior for x_shift parameter.
              Defaults to a Normal distribution.
            x_shift_constraint (Optional[Interval]): Constraint for x_shift
              parameter.
            x_shift_fixed (Union[float, torch.Tensor]): Provides the option to
              use a fixed parameter value. Defaults to None.

        Attributes:
            raw_x_shift (torch.nn.Parameter): Unconstrained parameter tensor
              of size x_dim.
            x_shift (torch.nn.Parameter): Constrained version of raw_x_shift.
        """
        super().__init__(model, **kwargs)
        x_shift_dim = kwargs.get("x_shift_dim", 1)
        self.register_parameter("raw_x_shift",
                                torch.nn.Parameter(torch.zeros(x_shift_dim)))
        x_shift_prior = kwargs.get(
            "x_shift_prior",
            NormalPrior(loc=torch.zeros((1, x_shift_dim)),
                        scale=torch.ones((1, x_shift_dim)))
        )
        if x_shift_prior is not None:
            self.register_prior("x_shift_prior", x_shift_prior,
                                self._x_shift_param, self._x_shift_closure)
        x_shift_constraint = kwargs.get("x_shift_constraint")
        if x_shift_constraint is not None:
            self.register_constraint("raw_x_shift", x_shift_constraint)
        # option to use a fixed parameter
        x_shift_fixed = kwargs.get("x_shift_fixed", None)
        if x_shift_fixed is not None:
            self.raw_x_shift.data = x_shift_fixed
            if x_shift_constraint is not None:
                raw_x_shift = self.raw_x_shift_constraint.inverse_transform(
                    torch.tensor(x_shift_fixed))
                self.raw_x_shift.data = raw_x_shift
            self.raw_x_shift.requires_grad = False

    @property
    def x_shift(self):
        return self._x_shift_param(self)

    @x_shift.setter
    def x_shift(self, value):
        self._x_shift_closure(self, value)

    def _x_shift_param(self, m):
        if hasattr(m, "raw_x_shift_constraint"):
            return m.raw_x_shift_constraint.transform(m.raw_x_shift)
        return m.raw_x_shift

    def _x_shift_closure(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_x_shift)
        if hasattr(m, "raw_x_shift_constraint"):
            m.initialize(
                raw_x_shift=m.raw_x_shift_constraint.inverse_transform(value))
        else:
            m.initialize(raw_x_shift=value)

    def input_offset_calibration(self, x):
        return x + self.x_shift

    def forward(self, x):
        return self.model(self.input_offset_calibration(x))


class InputScaleCalibration(CustomMean):
    def __init__(
            self,
            model: torch.nn.Module,
            **kwargs
    ):
        """Prior mean with learnable input scale calibration.

        Inputs are scaled by a learnable constant parameter:
        y = model(x_scale * x).

        Args:
            model: Inherited from CustomMean.

        Keyword Args:
            x_scale_dim (int): Dimension of the x_scale parameter.
              Defaults to 1.
            x_scale_prior (Optional[Prior]): Prior for x_scale parameter.
              Defaults to a Gamma distribution (concentration=1.0, rate=1.0).
            x_scale_constraint (Optional[Interval]): Constraint for x_scale
              parameter. Defaults to Positive().
            x_scale_fixed (Union[float, torch.Tensor]): Provides the option to
              use a fixed parameter value. Defaults to None.

        Attributes:
            raw_x_scale (torch.nn.Parameter): Unconstrained parameter tensor
              of size x_dim.
            x_scale (torch.nn.Parameter): Constrained version of raw_x_scale.
        """
        super().__init__(model, **kwargs)
        x_scale_dim = kwargs.get("x_scale_dim", 1)
        self.register_parameter("raw_x_scale",
                                torch.nn.Parameter(torch.ones(x_scale_dim)))
        # mean=1.0, std=1.0
        x_scale_prior = kwargs.get(
            "x_scale_prior",
            GammaPrior(concentration=1.0 * torch.ones((1, x_scale_dim)),
                       rate=1.0 * torch.ones((1, x_scale_dim)))
        )
        if x_scale_prior is not None:
            self.register_prior("x_scale_prior", x_scale_prior,
                                self._x_scale_param, self._x_scale_closure)
        x_scale_constraint = kwargs.get("x_scale_constraint", Positive())
        if x_scale_constraint is not None:
            self.register_constraint("raw_x_scale", x_scale_constraint)
            # correct initial value
            raw_x_scale_init = self.raw_x_scale_constraint.inverse_transform(
                torch.ones(x_scale_dim))
            self.raw_x_scale.data = raw_x_scale_init
        # option to use a fixed parameter
        x_scale_fixed = kwargs.get("x_scale_fixed", None)
        if x_scale_fixed is not None:
            self.raw_x_scale.data = x_scale_fixed
            if x_scale_constraint is not None:
                raw_x_scale = self.raw_x_scale_constraint.inverse_transform(
                    torch.tensor(x_scale_fixed))
                self.raw_x_scale.data = raw_x_scale
            self.raw_x_scale.requires_grad = False

    @property
    def x_scale(self):
        return self._x_scale_param(self)

    @x_scale.setter
    def x_scale(self, value):
        self._x_scale_closure(self, value)

    def _x_scale_param(self, m):
        if hasattr(m, "raw_x_scale_constraint"):
            return m.raw_x_scale_constraint.transform(m.raw_x_scale)
        return m.raw_x_scale

    def _x_scale_closure(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_x_scale)
        if hasattr(m, "raw_x_scale_constraint"):
            m.initialize(
                raw_x_scale=m.raw_x_scale_constraint.inverse_transform(value))
        else:
            m.initialize(raw_x_scale=value)

    def input_scale_calibration(self, x):
        return self.x_scale * x

    def forward(self, x):
        return self.model(self.input_scale_calibration(x))


class LinearInputCalibration(InputOffsetCalibration, InputScaleCalibration):
    def __init__(
            self,
            model: torch.nn.Module,
            x_dim: Optional[int] = None,
            **kwargs,
    ):
        """Prior mean with learnable linear input calibration.

        Inputs are passed through decoupled linear calibration nodes
        with constant learnable shift and scaling parameters:
        y = model(x_scale * (x + x_shift)).

        Args:
            model: Inherited from CustomMean.
            x_dim: Overwrites x_shift_dim and x_scale_dim.

        Keyword Args:
            Inherited from InputOffsetCalibration and InputScaleCalibration..

        Attributes:
            Inherited from InputOffsetCalibration and InputScaleCalibration.
        """
        if x_dim is not None:
            kwargs["x_shift_dim"] = x_dim
            kwargs["x_scale_dim"] = x_dim
        super().__init__(model, **kwargs)

    def linear_input_calibration(self, x):
        return self.input_scale_calibration(self.input_offset_calibration(x))

    def forward(self, x):
        return self.model(self.linear_input_calibration(x))


class OutputOffsetCalibration(CustomMean):
    def __init__(
            self,
            model: torch.nn.Module,
            **kwargs
    ):
        """Prior mean with learnable output offset calibration.

        Outputs are offset by a learnable constant parameter:
        y = model(x) + y_shift.

        Args:
            model: Inherited from CustomMean.

        Keyword Args:
            y_shift_dim: Dimension of the y_shift parameter. Defaults to 1.
            y_shift_prior (Optional[Prior]): Prior for y_shift parameter.
              Defaults to a Normal distribution.
            y_shift_constraint (Optional[Interval]): Constraint for y_shift
              parameter. Defaults to None.
            y_shift_fixed (Union[float, torch.Tensor]): Provides the option to
              use a fixed parameter value. Defaults to None.

        Attributes:
            raw_y_shift (torch.nn.Parameter): Unconstrained parameter tensor
              of size y_dim.
            y_shift (torch.nn.Parameter): Constrained version of raw_y_shift.
        """
        super().__init__(model, **kwargs)
        y_shift_dim = kwargs.get("y_shift_dim", 1)
        self.register_parameter("raw_y_shift",
                                torch.nn.Parameter(torch.zeros(y_shift_dim)))
        y_shift_prior = kwargs.get(
            "y_shift_prior",
            NormalPrior(loc=torch.zeros((1, y_shift_dim)),
                        scale=torch.ones((1, y_shift_dim)))
        )
        if y_shift_prior is not None:
            self.register_prior("y_shift_prior", y_shift_prior,
                                self._y_shift_param, self._y_shift_closure)
        y_shift_constraint = kwargs.get("y_shift_constraint")
        if y_shift_constraint is not None:
            self.register_constraint("raw_y_shift", y_shift_constraint)
        # option to use a fixed parameter
        y_shift_fixed = kwargs.get("y_shift_fixed", None)
        if y_shift_fixed is not None:
            self.raw_y_shift.data = y_shift_fixed
            if y_shift_constraint is not None:
                raw_y_shift = self.raw_y_shift_constraint.inverse_transform(
                    torch.tensor(y_shift_fixed))
                self.raw_y_shift.data = raw_y_shift
            self.raw_y_shift.requires_grad = False

    @property
    def y_shift(self):
        return self._y_shift_param(self)

    @y_shift.setter
    def y_shift(self, value):
        self._y_shift_closure(self, value)

    def _y_shift_param(self, m):
        if hasattr(m, "raw_y_shift_constraint"):
            return m.raw_y_shift_constraint.transform(m.raw_y_shift)
        return m.raw_y_shift

    def _y_shift_closure(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_y_shift)
        if hasattr(m, "raw_y_shift_constraint"):
            m.initialize(
                raw_y_shift=m.raw_y_shift_constraint.inverse_transform(value))
        else:
            m.initialize(raw_y_shift=value)

    def output_offset_calibration(self, y):
        return y + self.y_shift

    def forward(self, x):
        return self.output_offset_calibration(self.model(x))


class OutputScaleCalibration(CustomMean):
    def __init__(
            self,
            model: torch.nn.Module,
            **kwargs
    ):
        """Prior mean with learnable output scale calibration.

        Outputs are scaled by a learnable constant parameter:
        y = y_scale * model(x).

        Args:
            model: Inherited from CustomMean.

        Keyword Args:
            y_scale_dim: Dimension of the y_scale parameter. Defaults to 1.
            y_scale_prior (Optional[Prior]): Prior for y_scale parameter.
              Defaults to a Gamma distribution (concentration=1.0, rate=1.0).
            y_scale_constraint (Optional[Interval]): Constraint for y_scale
              parameter. Defaults to Positive().
            y_scale_fixed (Union[float, torch.Tensor]): Provides the option to
              use a fixed parameter value. Defaults to None.

        Attributes:
            raw_y_scale (torch.nn.Parameter): Unconstrained parameter tensor
              of size y_dim.
            y_scale (torch.nn.Parameter): Constrained version of raw_y_scale.
        """
        super().__init__(model, **kwargs)
        y_scale_dim = kwargs.get("y_scale_dim", 1)
        self.register_parameter("raw_y_scale",
                                torch.nn.Parameter(torch.ones(y_scale_dim)))
        # mean=1.0, std=1.0
        y_scale_prior = kwargs.get(
            "y_scale_prior",
            GammaPrior(concentration=1.0 * torch.ones((1, y_scale_dim)),
                       rate=1.0 * torch.ones((1, y_scale_dim)))
        )
        if y_scale_prior is not None:
            self.register_prior("y_scale_prior", y_scale_prior,
                                self._y_scale_param, self._y_scale_closure)
        y_scale_constraint = kwargs.get("y_scale_constraint", Positive())
        if y_scale_constraint is not None:
            self.register_constraint("raw_y_scale", y_scale_constraint)
            # correct initial value
            raw_y_scale_init = self.raw_y_scale_constraint.inverse_transform(
                torch.ones(y_scale_dim))
            self.raw_y_scale.data = raw_y_scale_init
        # option to use a fixed parameter
        y_scale_fixed = kwargs.get("y_scale_fixed", None)
        if y_scale_fixed is not None:
            self.raw_y_scale.data = y_scale_fixed
            if y_scale_fixed is not None:
                raw_y_scale = self.raw_y_scale_constraint.inverse_transform(
                    torch.tensor(y_scale_fixed))
                self.raw_y_scale.data = raw_y_scale
            self.raw_y_scale.requires_grad = False

    @property
    def y_scale(self):
        return self._y_scale_param(self)

    @y_scale.setter
    def y_scale(self, value):
        self._y_scale_closure(self, value)

    def _y_scale_param(self, m):
        if hasattr(m, "raw_y_scale_constraint"):
            return m.raw_y_scale_constraint.transform(m.raw_y_scale)
        return m.raw_y_scale

    def _y_scale_closure(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_y_scale)
        if hasattr(m, "raw_y_scale_constraint"):
            m.initialize(
                raw_y_scale=m.raw_y_scale_constraint.inverse_transform(value))
        else:
            m.initialize(raw_y_scale=value)

    def output_scale_calibration(self, y):
        return self.y_scale * y

    def forward(self, x):
        return self.output_scale_calibration(self.model(x))


class LinearOutputCalibration(OutputOffsetCalibration, OutputScaleCalibration):
    def __init__(
            self,
            model: torch.nn.Module,
            y_dim: Optional[int] = None,
            **kwargs,
    ):
        """Prior mean with learnable linear output calibration.

        Outputs are passed through decoupled linear calibration nodes
        with constant learnable shift and scaling parameters:
        y = y_scale * (model(x) + y_shift).

        Args:
            model: Inherited from CustomMean.
            y_dim: Overwrites y_shift_dim and y_scale_dim.

        Keyword Args:
            Inherited from OutputOffsetCalibration and OutputScaleCalibration.

        Attributes:
            Inherited from OutputOffsetCalibration and OutputScaleCalibration.
        """
        if y_dim is not None:
            kwargs["y_shift_dim"] = y_dim
            kwargs["y_scale_dim"] = y_dim
        super().__init__(model, **kwargs)

    def linear_output_calibration(self, y):
        return self.output_scale_calibration(self.output_offset_calibration(y))

    def forward(self, x):
        return self.linear_output_calibration(self.model(x))


class LinearCalibration(LinearInputCalibration, LinearOutputCalibration):
    def __init__(
            self,
            model: torch.nn.Module,
            **kwargs,
    ):
        """Prior mean with learnable linear in- and output calibration.

        In- and outputs are passed through decoupled linear calibration nodes
        with constant learnable shift and scaling parameters:
        y = y_scale * (model(x_scale * (x + x_shift)) + y_shift).

        Args:
            model: Inherited from CustomMean.

        Keyword Args:
            Inherited from LinearInputCalibration and LinearOutputCalibration.

        Attributes:
            Inherited from LinearInputCalibration and LinearOutputCalibration.
        """
        super().__init__(model, **kwargs)

    def forward(self, x):
        _x = self.linear_input_calibration(x)
        return self.linear_output_calibration(self.model(_x))
