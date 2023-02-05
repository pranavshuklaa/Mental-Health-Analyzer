// anger issues questionnaire
const questionnaire=[
    {

        question:"1. I was irritated more than people knew",
        a:"1",
        b:"2",
        c:"3",
        d:"4",
        e:"5"
    },
    {

        question:"2. I felt angry",
        a:"1",
        b:"2",
        c:"3",
        d:"4",
        e:"5"
    },
    {

        question:"3. I felt like I was ready to explode",
        a:"1",
        b:"2",
        c:"3",
        d:"4",
        e:"5"
    },
    {

        question:"4. I was grouchy.",
        a:"1",
        b:"2",
        c:"3",
        d:"4",
        e:"5"
    },
    {

        question:"5. I felt annoyed",
        a:"1",
        b:"2",
        c:"3",
        d:"4",
        e:"5"
    },
];

const question= document.querySelector('.question');
const option1=document.querySelector('#option1')
const option2=document.querySelector('#option2')
const option3=document.querySelector('#option3')
const option4=document.querySelector('#option4')
const option5=document.querySelector('#option5')
const next=document.querySelector('#next')
const answers=document.querySelectorAll('.answer')
const showScore=document.querySelector('#showScore')
const showProblem=document.querySelector('#showProblem')
const mainq=document.querySelector('#mainq')
const optionId=document.querySelector('#optionId')
const submit=document.querySelector('#submit')
const greet=document.querySelector('#greet')
const note=document.querySelector('#note')

let quesCount=0;
var score=0;
const loadQuestion=()=>{
    const questionList= questionnaire[quesCount]
    question.innerHTML=questionList.question;
    option1.innerHTML=questionList.a;
    option2.innerHTML=questionList.b;
    option3.innerHTML=questionList.c;
    option4.innerHTML=questionList.d;
    option5.innerHTML=questionList.e;
}

loadQuestion();

const getCheckAnswer=()=>{
    let answer;
    answers.forEach((currAnsElem)=>{
        if(currAnsElem.checked){
            answer=currAnsElem.id;
        }
    });
    return answer;
};

const deSelectAll=()=>{
    answers.forEach((currAnsElem)=>currAnsElem.checked=false)
}
// submit.disabled=true

next.addEventListener('click',()=>{
    // greet.innerHTML=``;
    // note.innerHTML=``;
    const checkedAnswer=getCheckAnswer();
    const result=console.log(checkedAnswer);

    if(checkedAnswer==1){
        score+=1
    }
    else if(checkedAnswer==2){
        score+=2
    }
    else if(checkedAnswer==3){
        score+=3
    }
    else if(checkedAnswer==4){
        score+=4
    }
    else if(checkedAnswer==5){
        score+=5
    }
    // else{
    //     score=score
    // }
    
    else{
        a=confirm('Do you want to skip this question?')
        if(a){
            loadQuestion();
        }
        else{
            quesCount-=1;
            loadQuestion();
            deSelectAll();
        }
    }
    score=score;
    quesCount++;
    deSelectAll();
    
    // else{
    //     a=confirm('Do you want to skip this question')
    //     if(a){
    //         alert('Question Skipped')
    //     }
    //     else{
    //         quesCount-=1;
    //         loadQuestion();
    //         deSelectAll();
    //     }
    // }
    
    if(quesCount<questionnaire.length-1){
        loadQuestion();
    }
    else if(quesCount<questionnaire.length){
        next.innerHTML=`<h3>Submit</h3>`
        loadQuestion();
        // submit.disabled=false;
        next.disabled=true;
        next.innerHTML=`<h3>End of Questions</h3>`
    }

}),

submit.addEventListener('click',()=>{
    // submit.addEventListener("hover",disable1);
    let confirmation=confirm('Do you want to submit?')
    if(confirmation){
    const checkedAnswer=getCheckAnswer();
    const result=console.log(checkedAnswer);
    
    if(checkedAnswer==1){
        score+=1
    }
    else if(checkedAnswer==2){
        score+=2
    }
    else if(checkedAnswer==3){
        score+=3
    }
    else if(checkedAnswer==4){
        score+=4
    }
    else if(checkedAnswer==5){
        score+=5
    }
    else{
        score=score;
    }
        //for blanking
        optionId.innerHTML=``
        mainq.innerHTML=``  
        quesId.innerHTML=``
        submit.disabled=true
        submit.innerHTML=`<h3>Test Submitted</h3>`
        next.disabled=true
        next.innerHTML=`<h3>
        Wohoo!, you did it! Congrats on completing the quiz and taking a big step towards improving your mental health. By clicking the "Solution" button, you'll be taken to a treasure trove of solutions tailored to your specific needs. Get ready to embark on a journey towards better mental wellness - good luck! 🚀
        
        </h3>
        `
        
        const val=score
        console.log(val)
        if(val>=0 && val<13){
            submit.innerHTML=`<h3>Severity level: None to slight</h3>`
        }
        else if(val>=13 && val<=16){
            submit.innerHTML=`<h3>Severity level: Mild </h3>`
        }
        else if(val>17 && val<=20){
            submit.innerHTML=`<h3>Severity level: Moderate </h3>`
        }
        else{
            submit.innerHTML=`<h3>Severity level: Severe</h3>`;
        }
    
        showProblem.innerHTML=`<button class="btn" onclick="location.reload()">Give Test Again </button> <br> <button class="btn" onclick="window.location.href='/solution3'">See Solutions</button> <br>`
        showProblem.classList.remove('problemArea')
    }
    else{
        alert('Quiz was not submitted')
    }

    }

);