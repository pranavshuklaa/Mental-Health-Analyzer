$(document).ready(function(){
    $("form").submit(function(event) {
      event.preventDefault();
      var input_data = $("input[name='question']").val();
      $.ajax({
        type: "POST",
        url: '/predict',
        data: {question: input_data},
        success: function(data) {
          $("#response").html(data.response);
        }
      });
    });
  });
  